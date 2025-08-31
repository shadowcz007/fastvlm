# FastVLM - 高性能视觉语言模型库

FastVLM 是一个基于 ONNX Runtime 的高性能视觉语言模型库，支持图像理解和文本生成。该库提供了简单易用的 API，可以轻松集成到 Rust 项目中。

## 特性

- 🚀 **高性能**: 基于 ONNX Runtime，支持 GPU 加速
- 🖼️ **图像理解**: 支持多种图像格式 (PNG, JPEG, WebP)
- 📝 **文本生成**: 根据图像内容生成描述性文本
- 🔧 **易于集成**: 简单的 API 设计，易于使用
- 📦 **自动下载**: 自动下载和管理模型文件
- 🧹 **资源管理**: 自动清理和资源释放

## 安装

在你的 `Cargo.toml` 中添加依赖：

```toml
[dependencies]
fastvlm = { git = "https://github.com/your-username/fastvlm" }
tokio = { version = "1", features = ["full"] }
```

## 快速开始

### 基本使用

```rust
use fastvlm::{FastVLMClient, FastVLMConfig};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // 1. 创建客户端
    let mut client = FastVLMClient::new();
    
    // 2. 配置模型
    let config = FastVLMConfig {
        max_response_length: 50,
        default_prompt: "描述这张图片".to_string(),
    };
    
    // 3. 初始化模型（使用 data/fastvlm 目录）
    client.initialize(Some("data/fastvlm"), config).await?;
    
    // 4. 分析图片
    let result = client.analyze_image_file(
        "path/to/your/image.jpg",
        Some("这张图片展示了什么？".to_string())
    ).await?;
    
    println!("分析结果: {}", result.text);
    
    // 5. 清理资源（可选，Drop trait 会自动处理）
    client.cleanup();
    
    Ok(())
}
```

### 使用自定义模型路径

```rust
use fastvlm::FastVLMClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = FastVLMClient::new();
    
    // 指定模型路径
    let config = FastVLMConfig::default();
    client.initialize(Some("/path/to/models"), config).await?;
    
    // 使用客户端...
    
    Ok(())
}
```

### 批量处理图片

```rust
use fastvlm::FastVLMClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = FastVLMClient::new();
    client.initialize(Some("data/fastvlm"), FastVLMConfig::default()).await?;
    
    let image_paths = vec![
        "image1.jpg".to_string(),
        "image2.png".to_string(),
        "image3.webp".to_string(),
    ];
    
    for path in image_paths {
        match client.analyze_image_file(&path, None).await {
            Ok(result) => println!("{}: {}", path, result.text),
            Err(e) => eprintln!("处理 {} 失败: {}", path, e),
        }
    }
    
    Ok(())
}
```

### 使用原始图像数据

```rust
use fastvlm::FastVLMClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = FastVLMClient::new();
    client.initialize(Some("data/fastvlm"), FastVLMConfig::default()).await?;
    
    // 读取图像文件
    let image_data = std::fs::read("image.jpg")?;
    
    // 使用 image crate 处理图像
    use image::io::Reader as ImageReader;
    let img = ImageReader::open("image.jpg")?.decode()?;
    let (width, height) = (img.width(), img.height());
    
    // 转换为 RGBA 字节
    let rgba_img = img.to_rgba8();
    let mut image_bytes = Vec::with_capacity((width * height * 4) as usize);
    
    for pixel in rgba_img.pixels() {
        image_bytes.extend_from_slice(&[pixel[0], pixel[1], pixel[2], pixel[3]]);
    }
    
    // 分析图像
    let result = client.analyze_image(
        image_bytes,
        width,
        height,
        Some("描述这张图片".to_string())
    ).await?;
    
    println!("分析结果: {}", result.text);
    
    Ok(())
}
```

## API 参考

### FastVLMClient

主要的客户端结构体，提供所有核心功能。

#### 方法

- `new() -> Self`: 创建新的客户端实例
- `initialize(model_path: Option<&str>, config: FastVLMConfig) -> Result<()>`: 初始化模型
- `analyze_image(image_data: Vec<u8>, width: u32, height: u32, prompt: Option<String>) -> Result<FastVLMAnalysisResult>`: 分析原始图像数据
- `analyze_image_file(image_path: &str, prompt: Option<String>) -> Result<FastVLMAnalysisResult>`: 从文件路径分析图像
- `is_initialized() -> bool`: 检查模型是否已初始化
- `get_model_path() -> Option<&str>`: 获取当前模型路径
- `cleanup()`: 卸载模型并释放资源

### FastVLMConfig

模型配置结构体。

```rust
pub struct FastVLMConfig {
    pub max_response_length: usize,  // 最大响应长度
    pub default_prompt: String,      // 默认提示文本
}
```

### FastVLMAnalysisResult

分析结果结构体。

```rust
pub struct FastVLMAnalysisResult {
    pub text: String,                    // 生成的文本
    pub timestamp: Instant,              // 时间戳
    pub processing_time: Duration,       // 处理时间
}
```

## 命令行工具

项目还包含一个命令行工具，可以直接使用：

```bash
# 编译
cargo build --release

# 分析单张图片
./target/release/fastvlm-cli image.jpg

# 批量分析多张图片
./target/release/fastvlm-cli image1.jpg image2.png image3.webp
```

## 模型文件

FastVLM 需要以下模型文件：

- `tokenizer.json`: 分词器
- `vision_encoder.onnx`: 视觉编码器
- `embed_tokens.onnx`: 词嵌入
- `decoder_model_merged.onnx`: 解码器

这些文件会在首次使用时自动下载到默认目录：
- 项目目录: `data/fastvlm/`（推荐）
- macOS: `~/Library/Application Support/fastvlm/`
- Linux: `~/.local/share/fastvlm/`
- Windows: `%APPDATA%\fastvlm\`

## 性能优化

### GPU 加速

- **macOS**: 自动使用 CoreML GPU 加速
- **Windows**: 自动使用 CUDA GPU 加速（如果可用）
- **Linux**: 使用 CPU 优化

### 内存管理

- 模型会在 `Drop` trait 中自动清理
- 可以手动调用 `cleanup()` 方法释放资源
- 支持并发使用（ONNX Runtime 是线程安全的）

## 错误处理

库使用 `anyhow` 进行错误处理，所有方法都返回 `Result<T, anyhow::Error>`：

```rust
match client.analyze_image_file("image.jpg", None).await {
    Ok(result) => println!("成功: {}", result.text),
    Err(e) => eprintln!("错误: {}", e),
}
```

## 示例项目

查看 `examples/` 目录中的完整示例：

- `basic_usage.rs`: 基本使用示例
- `batch_processing.rs`: 批量处理示例
- `custom_config.rs`: 自定义配置示例

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 更新日志

### v0.1.0
- 初始版本
- 支持基本的图像分析和文本生成
- 自动模型下载和管理
- 命令行工具


