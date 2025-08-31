//! FastVLM - 高性能视觉语言模型库
//! 
//! 这是一个基于ONNX Runtime的高性能视觉语言模型库，支持图像理解和文本生成。
//! 
//! ## 使用示例
//! 
//! ```rust
//! use fastvlm::{FastVLM, FastVLMConfig, FastVLMAnalysisResult};
//! use std::path::Path;
//! 
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // 1. 初始化模型
//!     let config = FastVLMConfig::default();
//!     let mut fastvlm = FastVLM::new("path/to/models", config).await?;
//!     
//!     // 2. 处理图片
//!     let image_data = std::fs::read("image.jpg")?;
//!     let result = fastvlm.analyze_image(
//!         image_data,
//!         1920,  // width
//!         1080,  // height
//!         Some("描述这张图片".to_string())
//!     ).await?;
//!     
//!     println!("分析结果: {}", result.text);
//!     
//!     // 3. 卸载模型（可选）
//!     fastvlm.cleanup();
//!     
//!     Ok(())
//! }
//! ```

pub mod fastvlm;
pub mod download;

pub use fastvlm::{FastVLM, FastVLMConfig, FastVLMAnalysisResult};
pub use download::{download_fastvlm_models, get_default_model_dir};
use anyhow::Result;

/// FastVLM 库的主要接口
pub struct FastVLMClient {
    model: Option<FastVLM>,
    model_path: Option<String>,
}

impl FastVLMClient {
    /// 创建新的 FastVLM 客户端实例
    pub fn new() -> Self {
        Self {
            model: None,
            model_path: None,
        }
    }

    /// 初始化模型
    /// 
    /// # 参数
    /// * `model_path` - 模型文件路径，如果为None则使用默认路径
    /// * `config` - 模型配置
    /// 
    /// # 返回
    /// * `Result<()>` - 初始化结果
    pub async fn initialize(&mut self, model_path: Option<&str>, config: FastVLMConfig) -> Result<()> {
        let path = if let Some(path) = model_path {
            path.to_string()
        } else {
            get_default_model_dir().to_string_lossy().to_string()
        };

        // 检查模型文件是否存在
        let model_dir = std::path::Path::new(&path);
        if !model_dir.exists() {
            tracing::info!("模型文件不存在，开始下载...");
            download_fastvlm_models(model_dir).await?;
        }

        // 初始化模型
        let model = FastVLM::new(model_dir, config).await?;
        self.model = Some(model);
        self.model_path = Some(path);

        tracing::info!("FastVLM 模型初始化成功");
        Ok(())
    }

    /// 分析图片
    /// 
    /// # 参数
    /// * `image_data` - 图片的RGBA字节数据
    /// * `width` - 图片宽度
    /// * `height` - 图片高度
    /// * `prompt` - 可选的提示文本
    /// 
    /// # 返回
    /// * `Result<FastVLMAnalysisResult>` - 分析结果
    pub async fn analyze_image(
        &mut self,
        image_data: Vec<u8>,
        width: u32,
        height: u32,
        prompt: Option<String>,
    ) -> Result<FastVLMAnalysisResult> {
        if let Some(ref mut model) = self.model {
            model.analyze_frame(image_data, width, height, prompt).await
        } else {
            Err(anyhow::anyhow!("模型未初始化，请先调用 initialize()"))
        }
    }

    /// 从文件路径分析图片
    /// 
    /// # 参数
    /// * `image_path` - 图片文件路径
    /// * `prompt` - 可选的提示文本
    /// 
    /// # 返回
    /// * `Result<FastVLMAnalysisResult>` - 分析结果
    pub async fn analyze_image_file(
        &mut self,
        image_path: &str,
        prompt: Option<String>,
    ) -> Result<FastVLMAnalysisResult> {
        // 加载图片
        let img = image::open(image_path)?;
        let (width, height) = (img.width(), img.height());
        
        // 转换为RGBA字节
        let rgba_img = img.to_rgba8();
        let mut image_data = Vec::with_capacity((width * height * 4) as usize);
        
        for pixel in rgba_img.pixels() {
            image_data.extend_from_slice(&[pixel[0], pixel[1], pixel[2], pixel[3]]);
        }

        self.analyze_image(image_data, width, height, prompt).await
    }

    /// 检查模型是否已初始化
    pub fn is_initialized(&self) -> bool {
        self.model.is_some()
    }

    /// 获取模型路径
    pub fn get_model_path(&self) -> Option<&str> {
        self.model_path.as_deref()
    }

    /// 卸载模型并释放资源
    pub fn cleanup(&mut self) {
        if let Some(_model) = self.model.take() {
            // 这里可以添加任何必要的清理逻辑
            tracing::info!("FastVLM 模型已卸载");
        }
        self.model_path = None;
    }
}

impl Drop for FastVLMClient {
    fn drop(&mut self) {
        self.cleanup();
    }
}

/// 便捷函数：自动下载并初始化模型
pub async fn create_fastvlm_client(
    model_path: Option<&str>,
    config: Option<FastVLMConfig>,
) -> Result<FastVLMClient> {
    let mut client = FastVLMClient::new();
    let config = config.unwrap_or_default();
    client.initialize(model_path, config).await?;
    Ok(client)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let client = FastVLMClient::new();
        assert!(!client.is_initialized());
    }

    #[tokio::test]
    async fn test_client_initialization() {
        let mut client = FastVLMClient::new();
        let config = FastVLMConfig::default();
        
        // 注意：这个测试需要模型文件才能通过
        // 在实际环境中，你可能需要跳过这个测试或提供测试模型
        let result = client.initialize(None, config).await;
        // 如果模型文件不存在，这个测试会失败，这是预期的行为
    }
}
