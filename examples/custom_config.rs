//! 自定义配置示例
//! 
//! 这个示例展示了如何使用不同的配置选项来定制 FastVLM 的行为。

use fastvlm::{FastVLMClient, FastVLMConfig};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🚀 FastVLM 自定义配置示例");
    
    // 1. 创建客户端
    let mut client = FastVLMClient::new();
    
    // 2. 配置不同的模型参数
    let configs = vec![
        ("简短描述", FastVLMConfig {
            max_response_length: 20,
            default_prompt: "用一句话描述这张图片".to_string(),
        }),
        ("详细描述", FastVLMConfig {
            max_response_length: 100,
            default_prompt: "详细描述这张图片的内容、场景、颜色和细节".to_string(),
        }),
        ("情感分析", FastVLMConfig {
            max_response_length: 50,
            default_prompt: "分析这张图片传达的情感和氛围".to_string(),
        }),
        ("物体识别", FastVLMConfig {
            max_response_length: 40,
            default_prompt: "识别这张图片中的主要物体和元素".to_string(),
        }),
    ];
    
    // 3. 获取图片路径
    let args: Vec<String> = std::env::args().collect();
    let image_path = if args.len() > 1 {
        &args[1]
    } else {
        println!("💡 提示: 运行此示例时可以提供图片路径作为参数");
        println!("   例如: cargo run --example custom_config image.jpg");
        println!("   将使用示例图片路径进行演示");
        "sample.jpg"
    };
    
    // 4. 使用不同配置分析同一张图片
    for (config_name, config) in configs {
        println!("\n🔧 使用配置: {}", config_name);
        println!("   • 最大响应长度: {}", config.max_response_length);
        println!("   • 默认提示: {}", config.default_prompt);
        
        // 重新初始化客户端（使用新配置和 data/fastvlm 目录）
        client.cleanup();
        client.initialize(Some("data/fastvlm"), config).await?;
        
        // 分析图片
        if std::path::Path::new(image_path).exists() {
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
            println!("⚠️  图片文件不存在: {}", image_path);
            println!("   这是预期的行为，因为示例图片不存在");
        }
    }
    
    // 5. 演示自定义提示词
    println!("\n🎯 演示自定义提示词");
    
    // 使用默认配置和 data/fastvlm 目录
    let default_config = FastVLMConfig::default();
    client.cleanup();
    client.initialize(Some("data/fastvlm"), default_config).await?;
    
    if std::path::Path::new(image_path).exists() {
        let custom_prompts = vec![
            "这张图片是什么风格的艺术作品？",
            "图片中有什么颜色？",
            "这张图片适合什么场合使用？",
            "图片中的人物在做什么？",
        ];
        
        for prompt in custom_prompts {
            println!("\n🤔 提示: {}", prompt);
            
            match client.analyze_image_file(image_path, Some(prompt.to_string())).await {
                Ok(result) => {
                    println!("📝 回答: {}", result.text);
                },
                Err(e) => {
                    println!("❌ 失败: {}", e);
                }
            }
        }
    }
    
    // 6. 演示模型路径配置
    println!("\n📁 演示模型路径配置");
    
    // 获取默认模型目录
    let default_model_dir = fastvlm::get_default_model_dir();
    println!("默认模型目录: {}", default_model_dir.display());
    
    // 检查模型文件是否存在
    let model_files = vec![
        "tokenizer.json",
        "vision_encoder.onnx",
        "embed_tokens.onnx",
        "decoder_model_merged.onnx",
    ];
    
    println!("模型文件检查:");
    for file in model_files {
        let file_path = default_model_dir.join(file);
        if file_path.exists() {
            println!("   ✅ {} - 存在", file);
        } else {
            println!("   ❌ {} - 不存在", file);
        }
    }
    
    // 7. 清理资源
    client.cleanup();
    println!("\n🧹 资源已清理");
    
    println!("\n📚 配置总结:");
    println!("• max_response_length: 控制生成文本的最大长度");
    println!("• default_prompt: 设置默认的提示词");
    println!("• 可以通过 analyze_image_file 的 prompt 参数覆盖默认提示");
    println!("• 模型路径可以通过 initialize 的第一个参数指定");
    
    Ok(())
}
