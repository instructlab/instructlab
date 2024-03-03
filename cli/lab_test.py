import lab

def test_convert(model_dir, adapter_file, quantized):
    print("tesing convert")
    lab.convert(model_dir, adapter_file, quantized)

def test_test(model_dir, adapter_file):
    print("tesing test")
    lab.test(model_dir, adapter_file)

def test_train(data_dir, model_dir, remote, quantize):
    print("tesing train")
    lab.train(data_dir, model_dir, remote, quantize)

if __name__ == "__main__":
    train_model_dir = "train/lora-mlx/ibm-merlinite-7b"
    model_dir = "train/lora-mlx/ibm-merlinite-7b-mlx-q"
    adapter_file = "train/lora-mlx/ibm-merlinite-7b-mlx-q/adapters-010.npz"
    quantized = True
    remote = False
    data_dir = "train/lora-mlx/data_puns"

    test_train(data_dir, train_model_dir, remote, quantized)
    # test_test(model_dir, adapter_file)
    # test_convert(model_dir, adapter_file, quantized)
    
