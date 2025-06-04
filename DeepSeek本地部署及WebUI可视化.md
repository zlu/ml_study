# DeepSeek本地部署及WebUI可视化教程

## 前言

DeepSeek是近年来备受关注的大模型之一，支持多种推理和微调场景。很多开发者希望在本地部署DeepSeek模型，并通过WebUI进行可视化交互。本文将详细介绍如何在本地环境下部署DeepSeek，并实现WebUI可视化，包括Ollama和CherryStudio的使用方法。

---

## 一、环境准备

### 1. 硬件要求
- 推荐NVIDIA显卡，显存16GB及以上（如A100、3090等）
- 至少50GB磁盘空间

### 2. 软件要求
- 操作系统：Linux或macOS（Windows建议使用WSL2）
- Python 3.8及以上
- CUDA 11.7及以上（如需GPU加速）
- pip最新版

---

## 二、安装依赖

```bash
# 更新pip
pip install --upgrade pip

# 安装PyTorch（根据你的CUDA版本选择）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# 安装DeepSeek相关依赖
pip install deepseek
```

---

## 三、下载DeepSeek模型

你可以通过HuggingFace或DeepSeek官方仓库下载模型权重。例如：

```bash
# 以DeepSeek LLM为例
git clone https://huggingface.co/deepseek-ai/deepseek-llm-7b-base
```

或者使用transformers库直接加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "deepseek-ai/deepseek-llm-7b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
```

---

## 四、部署WebUI可视化

### 1. 使用官方WebUI

DeepSeek官方或社区通常会提供基于Gradio或Streamlit的WebUI。以Gradio为例：

```bash
pip install gradio
```

创建`app.py`：

```python
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "deepseek-ai/deepseek-llm-7b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

def chat(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

iface = gr.Interface(fn=chat, inputs="text", outputs="text", title="DeepSeek WebUI")
iface.launch()
```

运行：

```bash
python app.py
```

浏览器访问 http://127.0.0.1:7860 即可体验。

---

### 2. 使用开源WebUI项目

你也可以使用如 [Open WebUI](https://github.com/open-webui/open-webui) 或 [text-generation-webui](https://github.com/oobabooga/text-generation-webui) 这类通用大模型WebUI，支持DeepSeek模型的加载和可视化。

---

## 五、常见问题

- **显存不足**：可尝试`torch_dtype="float16"`或`device_map="cpu"`，但速度会变慢。
- **模型下载慢**：建议使用国内镜像或提前下载模型文件。

---

## 六、使用Ollama和CherryStudio部署DeepSeek模型

### 1. Ollama本地部署DeepSeek

Ollama 是一个轻量级的大模型本地推理平台，支持一键拉取和运行多种主流大模型，包括DeepSeek。其优点是安装简单、界面友好、支持API调用。

#### 步骤如下：

1. **安装Ollama**
   - macOS 用户可直接在终端执行：
     ```bash
     brew install ollama
     ```
   - 或访问 [Ollama官网](https://ollama.com/) 下载适合你系统的安装包。

2. **拉取DeepSeek模型**
   Ollama官方模型库已支持DeepSeek（如未收录，可自定义导入模型权重）：
   ```bash
   ollama pull deepseek
   ```

3. **运行模型并启动本地服务**
   ```bash
   ollama run deepseek
   ```
   默认会在本地启动API服务，支持通过RESTful接口调用。

4. **WebUI可视化**
   - Ollama自带简易WebUI，访问 http://localhost:11434 即可体验。
   - 也可结合Gradio等工具自定义前端界面。

---

### 2. CherryStudio可视化管理DeepSeek

CherryStudio 是一个国产大模型可视化管理平台，支持多种大模型的本地/云端部署、微调和推理，界面友好，适合企业和个人开发者。

#### 使用步骤：

1. **注册并下载CherryStudio**
   - 访问 [CherryStudio官网](https://www.cherrystudio.cn/) 注册账号并下载客户端。

2. **安装并启动CherryStudio**
   - 按照安装向导完成部署，首次启动会自动检测本地环境。

3. **导入DeepSeek模型**
   - 在"模型管理"界面，选择"导入模型"，可选择本地已下载的DeepSeek权重，或通过HuggingFace链接自动下载。
   - 支持多种格式（如transformers、ggml等）。

4. **启动推理服务**
   - 在"推理服务"界面，选择DeepSeek模型，点击"一键部署"。
   - CherryStudio会自动分配端口并启动WebUI，支持多轮对话、参数调节等功能。

5. **WebUI体验**
   - 直接在CherryStudio客户端内体验，或通过分配的本地端口在浏览器访问。

---

## 七、总结

Ollama适合追求极简部署和API调用的开发者，CherryStudio则适合需要可视化管理和多模型协同的场景。两者都大大降低了本地部署大模型的门槛，让DeepSeek等大模型的本地应用变得更加便捷高效。

---

## 参考链接
- [DeepSeek官方HuggingFace页面](https://huggingface.co/deepseek-ai)
- [Gradio官方文档](https://gradio.app/)
- [text-generation-webui](https://github.com/oobabooga/text-generation-webui)
- [Ollama官网](https://ollama.com/)
- [CherryStudio官网](https://www.cherrystudio.cn/)

---

如需详细操作演示或遇到具体问题，欢迎留言交流！ 