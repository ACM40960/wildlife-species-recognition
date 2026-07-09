"""AI animal image recognition on night-vision camera-trap images (NTLNP).

Package modules:
    config      -- experiment configuration dataclass
    data        -- dataset loading, train/val/test splits, transforms
    model       -- transfer-learning model builder
    train       -- training / validation loop
    evaluate    -- metrics, confusion matrix, baseline comparison
    detect      -- optional YOLOv8 detect-and-crop stage
    utils       -- seeding, logging, plotting helpers
"""

__version__ = "0.1.0"
