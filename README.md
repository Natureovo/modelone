# 压缩图像梯度与频率变化分析

这个项目用于对比压缩前后图像在梯度域和频率域中的变化，定位细节损失、过平滑、块效应和高频退化区域，并输出可解释的热力图、候选增强 mask 和 JSON 报告。

## 运行

使用 Codex 自带 Python 运行时或本机 Python 均可，依赖只有 `numpy` 和 `Pillow`。

```powershell
python main.py --ref input\original.png --cmp input\compressed.jpg --out outputs --block 32
```

如果原图和压缩图尺寸不一致，程序会把压缩图 resize 到原图尺寸。

## 输出

`outputs/<case_name>/` 中会生成：

- `原图_亮度通道.png`：原图亮度通道
- `压缩图_亮度通道.png`：压缩图亮度通道
- `原图_梯度幅值.png` / `压缩图_梯度幅值.png`：梯度幅值图
- `梯度损失热力图.png`：梯度损失热力图
- `高频损失热力图.png`：高频损失热力图
- `块效应热力图.png`：块效应热力图
- `综合细节损失热力图.png`：综合细节损失热力图
- `候选增强区域_mask.png`：候选增强/生成区域 mask
- `综合细节损失叠加图.png`：综合损失叠加图
- `数值报告.json`：全局、区域统计，以及梯度/频率变化的具体数值
- `区域损失明细.csv`：每个图像块的区域级损失数值和区域建议

## 核心指标

综合细节损失分数：

```text
DetailLossScore =
0.35 * GradientLoss
+ 0.40 * HighFreqLoss
+ 0.15 * GradientDirectionChange
+ 0.10 * LocalVarianceLoss
```

分数越高，表示该区域越可能存在压缩造成的纹理或结构退化。

## 视频数据集

项目已包含 `dataset/` 目录，用于后续放置 HM 压缩视频测试数据：

```text
dataset/raw_videos/       原始视频或原始 YUV
dataset/hm_results/       HM 输出码流、重建 YUV 和日志
dataset/frame_pairs/      原始帧与 HM 压缩帧配对
dataset/metadata.csv      视频级元数据
dataset/frame_pairs.csv   帧级配对清单
```

后续用 HM 压缩后，建议把重建视频抽帧，与原始帧保持同名，再逐帧运行本项目。区域级判断以 `区域损失明细.csv` 为主，不只看整图平均值。

批量测试时，先填写 `dataset/frame_pairs.csv`，然后运行：

```powershell
python batch_analyze_dataset.py --dataset dataset --pairs frame_pairs.csv --out outputs\批量视频帧测试 --block 32
```

每个样本都会生成自己的热力图、区域明细和数值报告，最终总表为 `outputs\批量视频帧测试\批量汇总.csv`。

## 用 HM 构建视频测试数据集

1. 把原始视频放入 `dataset/raw_videos/`。
2. 准备 HM 的 `TAppEncoderStatic.exe` 和配置文件，例如 `encoder_randomaccess_main.cfg`。
3. 先生成 HM 输入 YUV、HM 命令和原始帧：

```powershell
python prepare_hm_dataset.py --dataset dataset --hm "D:\HM\bin\TAppEncoderStatic.exe" --hm-config "D:\HM\cfg\encoder_randomaccess_main.cfg" --qps 22,27,32,37,42 --frames 120 --frame-step 15
```

如果想让脚本直接调用 HM 编码，加上 `--run-hm`：

```powershell
python prepare_hm_dataset.py --dataset dataset --hm "D:\HM\bin\TAppEncoderStatic.exe" --hm-config "D:\HM\cfg\encoder_randomaccess_main.cfg" --qps 22,32,42 --frames 120 --frame-step 15 --run-hm
```

脚本会生成/更新：

```text
dataset/hm_results/input_yuv/      HM 输入 YUV
dataset/hm_results/bitstreams/     HM 码流
dataset/hm_results/recon_yuv/      HM 重建 YUV
dataset/hm_results/logs/           HM 日志
dataset/frame_pairs/               原始帧和压缩帧
dataset/metadata.csv               视频级元数据
dataset/frame_pairs.csv            帧级配对清单
```

如果暂时不加 `--run-hm`，HM 命令会写到 `dataset/hm_results/hm_commands.txt`，可以手动复制运行。等重建 YUV 生成后，再运行一次同样命令，脚本会抽取压缩帧并填充 `frame_pairs.csv`。
