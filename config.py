CFG = dict(
    # Data 
    train_color_dir = "data/train_img/color",
    train_depth_dir = "data/train_img/depth_generated",
    test_color_dir  = "data/test_img/color",
    test_depth_dir  = "data/test_img/depth_generated",
    image_size      = 256,
    batch_size      = 2,
    num_workers     = 8,
    balance_classes = True,       # WeightedRandomSampler on train set

    # Model 
    model_variant   = "base",     # "tiny" | "base" 
    num_classes     = 2,
    dropout         = 0.1,

    # Training 
    epochs          = 20,
    lr              = 1e-4,       
    weight_decay    = 1e-4,
    label_smoothing = 0.1,
    mixed_precision = True,       # AMP (set False if on CPU)

    #  Checkpointing 
    save_dir        = "checkpoints",
    save_every      = 5,          # also save a periodic checkpoint every N epochs

    # Early stopping 
    patience        = 10,         # epochs without improvement before stopping
    min_delta       = 1e-4,       # minimum improvement in val ACER to count
)