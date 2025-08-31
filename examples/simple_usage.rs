//! 简单使用示例
//! 
//! 这个示例展示了如何在其他 Rust 项目中使用 FastVLM 库。

use fastvlm::{FastVLMClient, FastVLMConfig};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🚀 FastVLM 库使用示例");
    
    // 创建客户端
    let mut client = FastVLMClient::new();
    
    // 配置模型
    let config = FastVLMConfig {
        max_response_length: 50,
        default_prompt: "描述这张图片".to_string(),
    };
    
    // 初始化模型（使用 data/fastvlm 目录）
    println!("🔧 正在初始化模型...");
    client.initialize(Some("data/fastvlm"), config).await?;
    println!("✅ 模型初始化成功");
    
    // 检查是否有图片文件参数
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        let image_path = &args[1];
        
        // 分析图片
        println!("📸 分析图片: {}", image_path);
        match client.analyze_image_file(image_path, None).await {
            Ok(result) => {
                println!("✅ 分析完成");
                println!("📝 结果: {}", result.text);
                println!("⏱️  处理时间: {:.2}秒", result.processing_time.as_secs_f32());
            },
            Err(e) => {
                println!("❌ 分析失败: {}", e);
            }
        }
    } else {
        println!("💡 提示: 运行此示例时可以提供图片路径作为参数");
        println!("   例如: cargo run --example simple_usage image.jpg");
    }
    
    // 清理资源
    client.cleanup();
    println!("🧹 资源已清理");
    
    Ok(())
}
