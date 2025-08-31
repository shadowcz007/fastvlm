//! 批量处理示例
//! 
//! 这个示例展示了如何使用 FastVLM 库进行批量图像处理。

use fastvlm::{FastVLMClient, FastVLMConfig};
use std::error::Error;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🚀 FastVLM 批量处理示例");
    
    // 1. 创建客户端并初始化
    let mut client = FastVLMClient::new();
    let config = FastVLMConfig {
        max_response_length: 30,
        default_prompt: "简要描述这张图片".to_string(),
    };
    
    println!("🔧 正在初始化模型...");
    client.initialize(Some("data/fastvlm"), config).await?;
    println!("✅ 模型初始化成功");
    
    // 2. 准备图片路径列表
    let args: Vec<String> = std::env::args().collect();
    let image_paths = if args.len() > 1 {
        args[1..].to_vec()
    } else {
        // 如果没有提供参数，使用示例图片路径
        vec![
            "sample1.jpg".to_string(),
            "sample2.png".to_string(),
            "sample3.webp".to_string(),
        ]
    };
    
    // 3. 验证图片文件是否存在
    let valid_paths: Vec<String> = image_paths
        .into_iter()
        .filter(|path| {
            if std::path::Path::new(path).exists() {
                true
            } else {
                println!("⚠️  文件不存在: {}", path);
                false
            }
        })
        .collect();
    
    if valid_paths.is_empty() {
        println!("❌ 没有找到有效的图片文件");
        println!("💡 提示: 运行此示例时可以提供图片路径作为参数");
        println!("   例如: cargo run --example batch_processing image1.jpg image2.png");
        return Ok(());
    }
    
    // 4. 开始批量处理
    let batch_start_time = Instant::now();
    println!("🚀 开始批量处理 {} 张图片", valid_paths.len());
    
    let mut success_count = 0;
    let mut total_processing_time = std::time::Duration::ZERO;
    
    for (i, image_path) in valid_paths.iter().enumerate() {
        println!("\n--- 处理第 {} 张图片: {} ---", i + 1, image_path);
        
        let image_start_time = Instant::now();
        match client.analyze_image_file(image_path, None).await {
            Ok(result) => {
                let processing_time = image_start_time.elapsed();
                total_processing_time += processing_time;
                success_count += 1;
                
                println!("✅ 分析成功");
                println!("📝 结果: {}", result.text);
                println!("⏱️  处理时间: {:.2}秒", processing_time.as_secs_f32());
            },
            Err(e) => {
                let processing_time = image_start_time.elapsed();
                println!("❌ 分析失败: {}", e);
                println!("⏱️  失败时间: {:.2}秒", processing_time.as_secs_f32());
            }
        }
    }
    
    // 5. 生成统计报告
    let total_time = batch_start_time.elapsed();
    let success_rate = if !valid_paths.is_empty() {
        (success_count as f32 / valid_paths.len() as f32) * 100.0
    } else {
        0.0
    };
    
    println!("\n📊 批量处理统计报告");
    println!("{}", "=".repeat(50));
    println!("📈 总体统计:");
    println!("   • 总图片数量: {}", valid_paths.len());
    println!("   • 成功处理: {} 张", success_count);
    println!("   • 处理失败: {} 张", valid_paths.len() - success_count);
    println!("   • 成功率: {:.1}%", success_rate);
    
    if success_count > 0 {
        let avg_time = total_processing_time / success_count as u32;
        println!("\n⏱️  时间统计:");
        println!("   • 总处理时间: {:.2}秒", total_processing_time.as_secs_f32());
        println!("   • 平均处理时间: {:.2}秒", avg_time.as_secs_f32());
        println!("   • 总耗时: {:.2}秒", total_time.as_secs_f32());
    }
    
    println!("{}", "=".repeat(50));
    
    // 6. 清理资源
    client.cleanup();
    println!("🧹 资源已清理");
    
    Ok(())
}
