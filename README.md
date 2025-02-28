# ml_tutorial
Introduction to Flordb with PyTorch

## Hindsight Logging


Whenever something looks wrong during training, you can use FlorDB to replay the previous runs and log additional information, like the gradient norm. To log the gradient norm, you can add the following line to the training script:

```python
flor.log("gradient_norm", 
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=float('inf')
    ).item()
)
```