import numpy as np

def dequantize_fp8_e4m3(fp8_weights):
    """
    Convert FP8 e4m3 format back to float32
    e4m3: 1 sign bit + 4 exponent bits + 3 mantissa bits
    """
    # Extract bit components
    sign = (fp8_weights >> 7) & 0x1
    exponent = (fp8_weights >> 3) & 0xF  # 4 bits
    mantissa = fp8_weights & 0x7  # 3 bits
    
    # Handle special cases
    if exponent == 0:
        if mantissa == 0:
            # Zero
            result = 0.0
        else:
            # Subnormal numbers
            result = (-1)**sign * (mantissa / 8.0) * (2**(-6))
    elif exponent == 15:  # All 1s in exponent
        # NaN or Infinity
        result = float('nan') if mantissa != 0 else float('inf')
        if sign:
            result = -result
    else:
        # Normal numbers
        # Bias for e4m3 is 7 (2^(4-1) - 1)
        actual_exponent = exponent - 7
        mantissa_value = 1.0 + (mantissa / 8.0)  # Add implicit leading 1
        result = (-1)**sign * mantissa_value * (2**actual_exponent)
    
    return result

def dequantize_weight_blocks(quantized_tensor, block_size=(128, 128)):
    """
    Dequantize weights that were quantized in blocks
    """
    rows, cols = quantized_tensor.shape
    dequantized = np.zeros((rows, cols), dtype=np.float32)
    
    block_rows, block_cols = block_size
    
    for i in range(0, rows, block_rows):
        for j in range(0, cols, block_cols):
            # Extract block
            end_i = min(i + block_rows, rows)
            end_j = min(j + block_cols, cols)
            block = quantized_tensor[i:end_i, j:end_j]
            
            # Dequantize block
            dequantized_block = np.vectorize(dequantize_fp8_e4m3)(block)
            dequantized[i:end_i, j:end_j] = dequantized_block
    
    return dequantized

from safetensors import safe_open

def dequantize_model_weights(model_path):
    """
    Load and dequantize all model weights
    """
    dequantized_weights = {}
    
    # Load each safetensors file
    for i in range(1, 5):  # model-00001 to model-00004
        file_path = f"{model_path}/model-{i:05d}-of-00004.safetensors"
        
        with safe_open(file_path, framework="numpy") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                
                # Check if this layer should be dequantized
                if should_dequantize_layer(key):
                    dequantized_weights[key] = dequantize_weight_blocks(tensor)
                else:
                    # Keep as-is (these are in modules_to_not_convert)
                    dequantized_weights[key] = tensor
                    
    return dequantized_weights

def should_dequantize_layer(layer_name):
    """
    Check if layer should be dequantized based on modules_to_not_convert
    """
    modules_to_not_convert = [
        "lm_head",
        # All the layer norms and gates from config...
    ]
    
    for module in modules_to_not_convert:
        if module in layer_name:
            return False
    return True

