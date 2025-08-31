# FastVLM - 高性能图片 AI 分析工具

一个基于 FastVLM 模型的图片分析工具，支持批量处理图片并生成中文描述，具备完整的性能监控和时间统计功能。

## ✨ 功能特性

- 🚀 **高性能推理**: 基于 ONNX Runtime 和 CoreML (macOS) / CUDA (Windows) 加速
- 📸 **批量处理**: 支持同时处理多张图片，自动验证文件存在性
- 🎯 **中文输出**: 生成中文图片描述，支持自定义提示词
- ⏱️ **详细统计**: 提供完整的处理时间统计和分析报告
- 🔧 **自动下载**: 首次运行时自动下载模型文件
- 📊 **性能监控**: 实时监控初始化、预处理、推理各阶段耗时
- 🛠️ **错误处理**: 完善的错误处理和失败统计

## 🏗️ 技术架构

### 核心组件
- **FastVLM**: 基于 ONNX 的视觉语言模型
- **图像处理器**: 预处理和特征提取
- **文本生成器**: 基于 Transformer 的解码器
- **时间统计器**: 详细的性能监控和分析
- **模型下载器**: 自动模型管理和下载

### 处理流程
1. **初始化阶段**: 加载模型和 tokenizer，统计加载时间
2. **图像预处理**: 调整大小和格式转换
3. **特征提取**: 使用视觉编码器提取图像特征
4. **文本生成**: 融合图像和文本特征生成描述
5. **统计报告**: 生成详细的性能分析报告

## 📦 安装和使用

### 系统要求
- **操作系统**: macOS 或 Windows
- **Rust**: 1.70+ 
- **内存**: 至少 4GB 可用内存
- **存储**: 约 1.4GB 磁盘空间（用于模型下载）

### 编译安装
```bash
# 克隆项目
git clone <repository-url>
cd fastvlm

# 编译发布版本
cargo build --release

# 编译调试版本（可选）
cargo build
```

### 基本使用
```bash
# 处理单张图片
./target/release/fastvlm image.jpg

# 批量处理多张图片
./target/release/fastvlm image1.jpg image2.png image3.webp

# 查看帮助
./target/release/fastvlm
```

## 📊 时间统计功能

### 初始化时间统计
程序启动时会显示详细的初始化统计信息：

```
🔧 初始化 FastVLM...
🔍 搜索 FastVLM 模型...
找到 FastVLM 数据目录: /path/to/models
Tokenizer loaded in 45.23ms
Model vision_encoder.onnx loaded in 234.56ms
Model embed_tokens.onnx loaded in 123.45ms
Model decoder_model_merged.onnx loaded in 567.89ms
FastVLM models loaded successfully in 971.13ms
✅ FastVLM 初始化成功！耗时: 1.23秒
```

### 图片处理时间统计
每张图片处理完成后会显示详细的时间信息：

```
📸 加载图片: image.jpg
✅ 图片加载成功: 1920x1080
🔄 调整图片大小到: 960x540
🤖 开始 AI 分析...
Image preprocessing completed in 12.34ms
Text generation completed in 2345.67ms
FastVLM analysis completed in 2358.01ms (preprocess: 12.34ms, generation: 2345.67ms): 这是一张风景照片
✅ 分析完成！耗时: 2.36秒
⏱️  单张图片处理时间: 2.36秒
```

### 批量处理统计报告
批量处理完成后会显示完整的统计报告：

```
📊 处理统计报告
==================================================
📈 总体统计:
   • 总图片数量: 3
   • 成功处理: 3 张
   • 处理失败: 0 张
   • 成功率: 100.0%

⏱️  时间统计:
   • 总处理时间: 15.23秒
   • 平均处理时间: 5.08秒
   • 最快处理时间: 4.85秒
   • 最慢处理时间: 5.45秒

📋 详细时间列表:
   • 图片 1: 5.12秒
   • 图片 2: 4.85秒
   • 图片 3: 5.45秒
==================================================
🔧 初始化耗时: 2.34秒
⏱️  总耗时 (初始化 + 处理): 17.57秒
```

## ⚙️ 配置选项

### FastVLMConfig 配置
```rust
pub struct FastVLMConfig {
    pub max_response_length: usize,  // 最大响应长度 (默认: 30)
    pub default_prompt: String,      // 默认提示词
}
```

### 自定义配置示例
```rust
let config = FastVLMConfig {
    max_response_length: 50,
    default_prompt: "请详细描述这张图片的内容和场景".to_string(),
};
```

### 应用配置
```rust
struct FastVLMApp {
    fastvlm: Option<fastvlm::FastVLM>,
    llm_prompt: String,              // 自定义提示词
    llm_resolution_scale: f32,       // 分辨率缩放比例 (默认: 0.5)
    init_time: Option<Duration>,     // 初始化时间
}
```

## 🔧 性能优化

### 分辨率缩放
默认将图片分辨率缩放至 50% 以提高处理速度：
```rust
llm_resolution_scale: 0.5, // 可调整为 0.3-0.7
```

### 模型优化
- 使用 ONNX Runtime 的 Level3 优化
- 支持 CoreML GPU 加速 (macOS)
- 支持 CUDA 加速 (Windows)
- 静态输入形状优化

### 内存优化
- 自动调整图片大小减少内存占用
- 流式处理避免内存溢出
- 及时释放中间结果

## 🐛 故障排除

### 常见问题

#### 1. 模型下载失败
```bash
# 检查网络连接
ping huggingface.co

# 手动下载模型
# 模型文件位置: ~/.cache/fastvlm/
```

#### 2. 内存不足
```bash
# 减少批量处理数量
./target/release/fastvlm image1.jpg image2.jpg  # 一次处理2张

# 降低分辨率缩放比例
# 修改 main.rs 中的 llm_resolution_scale 为 0.3
```

#### 3. 处理速度慢
```bash
# 检查 GPU 加速是否启用
export RUST_LOG=info
./target/release/fastvlm image.jpg

# 查看详细日志
export RUST_LOG=debug
./target/release/fastvlm image.jpg
```

### 日志级别
```bash
# 设置日志级别
export RUST_LOG=debug  # 详细日志
export RUST_LOG=info   # 信息日志
export RUST_LOG=warn   # 警告日志
export RUST_LOG=error  # 错误日志
```

## 📁 项目结构

```
fastvlm/
├── src/
│   ├── main.rs                    # 主程序入口
│   └── fastvlm/
│       ├── mod.rs                 # 模块定义
│       ├── fastvlm.rs             # 核心 FastVLM 实现
│       ├── download.rs            # 模型下载管理
│       └── fastvlm_image_process.rs # 图像预处理
├── data/                          # 模型文件目录
├── target/                        # 编译输出目录
├── Cargo.toml                     # 项目配置
├── Cargo.lock                     # 依赖锁定文件
└── README.md                      # 项目文档
```

## 🔧 开发指南

### 添加新功能
1. 在 `fastvlm.rs` 中添加核心功能
2. 在 `main.rs` 中添加用户界面
3. 更新统计功能以包含新的性能指标

### 测试
```bash
# 运行测试
cargo test

# 检查代码
cargo check

# 格式化代码
cargo fmt

# 代码检查
cargo clippy
```

### 性能基准测试
```bash
# 创建测试图片
convert -size 800x600 xc:lightblue test_image.jpg

# 运行性能测试
time ./target/release/fastvlm test_image.jpg
```

## 📈 性能基准

### 硬件配置
- **macOS M3**: ~2-3 秒推理时间
- **Windows RTX 4090**: ~1-2 秒推理时间
- **CPU 模式**: ~5-8 秒推理时间

### 内存使用
- **模型加载**: ~2GB 内存
- **单张图片处理**: ~500MB 内存
- **批量处理**: 根据图片数量线性增长

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [FastVLM](https://huggingface.co/onnx-community/FastVLM-0.5B-ONNX) - 核心 AI 模型
- [ONNX Runtime](https://onnxruntime.ai/) - 推理引擎
- [Rust](https://www.rust-lang.org/) - 编程语言
- [CoreML](https://developer.apple.com/machine-learning/) - macOS 机器学习框架
- [CUDA](https://developer.nvidia.com/cuda-zone) - NVIDIA GPU 计算平台

## 📞 支持

如果您遇到问题或有建议，请：
1. 查看 [故障排除](#故障排除) 部分
2. 搜索现有的 [Issues](../../issues)
3. 创建新的 Issue 并详细描述问题

---

**FastVLM** - 让图片分析更智能、更高效！ 🚀


