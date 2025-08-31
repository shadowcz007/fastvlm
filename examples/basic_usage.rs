//! 基本使用示例
//! 
//! 这个示例展示了如何使用 FastVLM 库进行基本的图像分析。

use fastvlm::{FastVLMClient, FastVLMConfig};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🚀 FastVLM 基本使用示例");
    
    // 1. 创建客户端
    let mut client = FastVLMClient::new();
    println!("✅ 客户端创建成功");
    
    // 2. 配置模型
    let config = FastVLMConfig {
        max_response_length: 50,
        default_prompt: "用中文描述这张图片的内容".to_string(),
    };
    println!("✅ 配置设置完成");
    
    // 3. 初始化模型（使用 data/fastvlm 目录）
    println!("🔧 正在初始化模型...");
    client.initialize(Some("data/fastvlm"), config).await?;
    println!("✅ 模型初始化成功");
    
    // 4. 检查模型状态
    if client.is_initialized() {
        println!("✅ 模型已就绪");
        if let Some(path) = client.get_model_path() {
            println!("📁 模型路径: {}", path);
        }
    } else {
        println!("❌ 模型未初始化");
        return Ok(());
    }
    
    // 5. 分析图片（如果提供了图片路径）
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        let image_path = &args[1];
        println!("📸 分析图片: {}", image_path);
        
        match client.analyze_image_file(image_path, None).await {
            Ok(result) => {
                println!("✅ 分析完成！");
                println!("📝 结果: {}", result.text);
                println!("⏱️  处理时间: {:.2}秒", result.processing_time.as_secs_f32());
            },
            Err(e) => {
                println!("❌ 分析失败: {}", e);
            }
        }
    } else {
        println!("💡 提示: 运行此示例时可以提供图片路径作为参数");
        println!("   例如: cargo run --example basic_usage image.jpg");
    }
    
    // 6. 清理资源
    client.cleanup();
    println!("🧹 资源已清理");
    
    Ok(())
}
