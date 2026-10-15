(** Loads a fused VLM checkpoint (safetensors) into [Vlm.model].

    The trained artifact is three models in one file — this is the shape
    of the thing being loaded, not a convenience:

    | namespace | contents | loaded by |
    | :--- | :--- | :--- |
    | [vit.] | the vision encoder | [Vit_loader] |
    | [adapter.] | the ViT→LLM projection | here |
    | [gpt.] | the decoder (`llm_quest.gpt.GPTModel`) | [Gpt_repo_loader] |

    [vlm_training.py] writes it as *two* PyTorch `.pth` pickles
    ([config.vlm_gpt] and [config.vlm_adapter]) rather than one
    safetensors file. This port reads safetensors only — it has no pickle
    reader and no plan for one, since a pickle is executable data and the
    rest of the loader story is "parse a documented, versioned binary
    format". Converting those two `.pth` files into this namespace layout
    is a one-line `safetensors.torch.save_file` away on the Python side
    and is recorded in ROADMAP rather than papered over.

    **Adapter layout is the one thing worth stating carefully.** [ViTAdapter]
    has two branches and they are not the same shape:
    ["simple" -> Linear(vit_d_out, llm_d_in)]
    ["ffn"    -> Linear(vit_d_out, vit_d_out*hidden_size_factor), GELU,
                 Linear(vit_d_out*hidden_size_factor, llm_d_in)]

    The middle activation is [torch.nn.GELU], i.e. the **exact** erf form,
    matching [Norm.Gelu_exact] — so the adapter's activation is not a free
    parameter and this loader refuses to accept one that disagrees. The
    dropout in the "ffn" branch is identity at inference (every shipped
    config sets it to 0.0), so it stores no tensor and there is nothing to
    load. *)

(** [ViTAdapter]'s linears are **bias-free** — [bias=False] is its default
    and both VLM scripts construct it with that default, so the real
    checkpoint has no [*.bias] here at all. Asserting their absence is not
    pedantry: an adapter that loaded a bias when the file had none would
    read garbage as [0.0]-shaped memory rather than fail, and an adapter
    that skipped a present one would silently change the projection.

    Key names are [ViTAdapter]'s own: [adapter.weight] for the simple
    branch, [adapter.0.weight] / [adapter.3.weight] for the ffn branch.

    **The index is 3, not 2,** because [ViTAdapter]'s ffn branch is
    [nn.Sequential(Linear, GELU, Dropout|Identity, Linear)] — there are
    four children, not three, because the dropout slot is there even when
    it holds an [nn.Identity]. Neither [nn.GELU] nor [nn.Dropout] nor
    [nn.Identity] carries a parameter, so the *keys* are [0.weight] and
    [3.weight] either way, and the gap in between is invisible in the
    checkpoint. With [Safetensors.subset ~prefix:"adapter."] they arrive
    here as [0.weight] and [3.weight] once the [adapter.] namespace has
    been sliced off (see [load], which has to take two of them).

    Read that index from the module rather than from the formula: the
    first draft of this loader claimed [2.weight], which is the position
    you would predict if you counted only the layers with weights. It is
    caught the moment the loader meets a real `state_dict`, which is the
    argument for exporting one. *)
let adapter_expected_keys ~adapter_type ~bias ~hidden_size_factor : string list =
  let lin n = if bias then [ Printf.sprintf "%d.weight" n; Printf.sprintf "%d.bias" n ]
              else [ Printf.sprintf "%d.weight" n ] in
  match adapter_type with `Simple -> lin 0 | `Ffn _ -> lin 0 @ lin 3

let load_adapter (f : Safetensors.file) ~adapter_type ~bias ~vit_d_out ~llm_d_in ~hidden_size_factor :
    Vlm.adapter =
  let lin n ~d_in ~d_out =
    {
      Attention.w =
        Safetensors.get_tensor f (Printf.sprintf "%d.weight" n) ~expected_shape:[ d_out; d_in ];
      b =
        (if bias then Some (Safetensors.get_vector f (Printf.sprintf "%d.bias" n) ~expected_len:d_out)
         else None);
      d_in;
      d_out;
    }
  in
  (* Branch first, then read. Both branches own an `0.weight`, so reading it
     before deciding would pick one branch's expected shape and fail on
     the other — including where the shapes coincide and the two adapters
     are numerically different anyway. *)
  match adapter_type with
  | `Simple -> Vlm.Simple (lin 0 ~d_in:vit_d_out ~d_out:llm_d_in)
  | `Ffn _ ->
      let inner = vit_d_out * hidden_size_factor in
      Vlm.Ffn
      {
        l1 = lin 0 ~d_in:vit_d_out ~d_out:inner;
        l2 = lin 3 ~d_in:inner ~d_out:llm_d_in;
        activation = Norm.Gelu_exact;
      }

(** The whole fused model.

    [~adapter_type] must be passed rather than inferred: both branches are
    a single [l0] Linear under the same key name, and the file alone
    cannot say which one it is. Guessing from a shape match would work for
    a checkpoint whose widths differ and silently pick the wrong read for
    one where they do not. *)
let load (f : Safetensors.file) ~(vit_cfg : Vit.t) ~(gpt_cfg : Config.t)
    ~(adapter_type : [ `Simple | `Ffn of int ]) ~(bias : bool) : Vlm.model =
  let vit = Vit_loader.load (Safetensors.subset f ~prefix:"vit.") vit_cfg in
  (* [Gpt_repo_loader], not [Gpt2_loader]: `vlm_training.py` builds this
     decoder from `llm_quest.gpt.GPTModel` and saves *its* state_dict, so
     the keys are [emb_dict.weight] / [trf_blocks.N....] — a third naming
     scheme alongside HF gpt2 and HF Llama. See [Gpt_repo_loader]'s header
     for the map. *)
  let gpt = Gpt_repo_loader.load (Safetensors.subset f ~prefix:"gpt.") gpt_cfg in
  let vit_d_out = vit_cfg.Vit.emb_dim in
  let llm_d_in = gpt_cfg.Config.hidden_size in
  let hidden_size_factor = match adapter_type with `Simple -> 1 | `Ffn h -> h in
  if hidden_size_factor <= 0 then invalid_arg "Vlm_loader: hidden_size_factor must be positive";
  (* The doubled prefix is not a mistake. [ViTAdapter]'s module attribute
     is itself named [adapter], so its own [state_dict] keys are
     [adapter.0.weight] / [adapter.3.weight]; under a file namespace
     called [adapter.] that lands as [adapter.adapter.0.weight]. Slicing
     through both is what makes the loader read the real names, which is
     the entire reason this export exists — a "tidied" name here would be
     a name no checkpoint has. *)
  let af = Safetensors.subset f ~prefix:"adapter.adapter." in
  Safetensors.assert_exact_keys af ~skip_prefix:"" (adapter_expected_keys ~adapter_type ~bias ~hidden_size_factor);
  let adapter = load_adapter af ~adapter_type ~bias ~vit_d_out ~llm_d_in ~hidden_size_factor in
  (* The adapter exists to make the two widths agree. If it does not, the
     fusion in [Vlm.fuse] raises deep inside a forward pass; catching it
     here names both widths instead. *)
  let out_width =
    match adapter with Vlm.Simple a -> a.Attention.d_out | Vlm.Ffn a -> a.l2.Attention.d_out
  in
  if out_width <> llm_d_in then
    failwith
      (Printf.sprintf "Vlm_loader: adapter projects to %d but the decoder expects %d" out_width
         llm_d_in);
  { Vlm.vit; adapter; gpt }