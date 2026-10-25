import torch

def print_weights(model):
    for name, param in model.items():
        print(f"{name}: {param.float().detach().cpu().numpy()}")

def save_weights_txt(model, out_path):
    with open(out_path, "w") as f:
        for name, param in model.items():
            f.write(f"{name}: {param.float().detach().cpu().numpy()}\n")

if __name__ == "__main__":
    # 通过 torch.load 加载你的模型权重文件
    model = torch.load("/primus_xpfs_workspace_T06/zhangyuxuan/projects1/LA_stage2_0318_womotion/logs_uniformnoise_6attn/ckpt/step_500_critic_lora.pt", map_location="cpu")
    print_weights(model)
    save_path = "./critic_model.txt"
    save_weights_txt(model,save_path)