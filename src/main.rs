use std::time::Instant;
use anyhow::Result;
use image::DynamicImage;

mod fastvlm;

#[derive(Debug, Clone)]
pub struct ImageAnalysisResult {
    pub text: String,
    pub processing_time: std::time::Duration,
    pub timestamp: Instant,
}

#[derive(Debug)]
pub struct ProcessingStats {
    pub total_images: usize,
    pub successful_images: usize,
    pub failed_images: usize,
    pub total_processing_time: std::time::Duration,
    pub average_processing_time: std::time::Duration,
    pub min_processing_time: std::time::Duration,
    pub max_processing_time: std::time::Duration,
    pub individual_times: Vec<std::time::Duration>,
}

impl ProcessingStats {
    fn new() -> Self {
        Self {
            total_images: 0,
            successful_images: 0,
            failed_images: 0,
            total_processing_time: std::time::Duration::ZERO,
            average_processing_time: std::time::Duration::ZERO,
            min_processing_time: std::time::Duration::MAX,
            max_processing_time: std::time::Duration::ZERO,
            individual_times: Vec::new(),
        }
    }

    fn add_result(&mut self, processing_time: std::time::Duration, success: bool) {
        self.total_images += 1;
        if success {
            self.successful_images += 1;
            self.total_processing_time += processing_time;
            self.individual_times.push(processing_time);
            
            if processing_time < self.min_processing_time {
                self.min_processing_time = processing_time;
            }
            if processing_time > self.max_processing_time {
                self.max_processing_time = processing_time;
            }
        } else {
            self.failed_images += 1;
        }
    }

    fn finalize(&mut self) {
        if self.successful_images > 0 {
            self.average_processing_time = self.total_processing_time / self.successful_images as u32;
        }
    }

    fn print_summary(&self) {
        println!("\n📊 处理统计报告");
        println!("{}", "=".repeat(50));
        println!("📈 总体统计:");
        println!("   • 总图片数量: {}", self.total_images);
        println!("   • 成功处理: {} 张", self.successful_images);
        println!("   • 处理失败: {} 张", self.failed_images);
        println!("   • 成功率: {:.1}%", (self.successful_images as f32 / self.total_images as f32) * 100.0);
        
        if self.successful_images > 0 {
            println!("\n⏱️  时间统计:");
            println!("   • 总处理时间: {:.2}秒", self.total_processing_time.as_secs_f32());
            println!("   • 平均处理时间: {:.2}秒", self.average_processing_time.as_secs_f32());
            println!("   • 最快处理时间: {:.2}秒", self.min_processing_time.as_secs_f32());
            println!("   • 最慢处理时间: {:.2}秒", self.max_processing_time.as_secs_f32());
            
            println!("\n📋 详细时间列表:");
            for (i, duration) in self.individual_times.iter().enumerate() {
                println!("   • 图片 {}: {:.2}秒", i + 1, duration.as_secs_f32());
            }
        }
        println!("{}", "=".repeat(50));
    }
}

struct FastVLMApp {
    fastvlm: Option<fastvlm::FastVLM>,
    llm_prompt: String,
    llm_resolution_scale: f32,
    init_time: Option<std::time::Duration>,
}

impl FastVLMApp {
    fn new() -> Result<Self> {
        let init_start_time = Instant::now();
        println!("🔧 初始化 FastVLM...");
        
        // 初始化 FastVLM
        let fastvlm = Self::init_fastvlm();
        let llm_enabled = fastvlm.is_some();
        
        let init_time = init_start_time.elapsed();
        
        if llm_enabled {
            println!("✅ FastVLM 初始化成功！耗时: {:.2}秒", init_time.as_secs_f32());
        } else {
            println!("⚠️  FastVLM 初始化失败 - AI 功能已禁用 (耗时: {:.2}秒)", init_time.as_secs_f32());
        }
        
        Ok(Self {
            fastvlm,
            llm_prompt: "用中文描述这张图片的内容".to_string(),
            llm_resolution_scale: 0.5, // 降低分辨率以提高处理速度
            init_time: Some(init_time),
        })
    }
    
    fn init_fastvlm() -> Option<fastvlm::FastVLM> {
        println!("🔍 搜索 FastVLM 模型...");
        
        // 获取系统标准模型目录
        let system_model_dir = fastvlm::download::get_default_model_dir();
        
        let possible_paths = [
            system_model_dir.as_path(),
            std::path::Path::new("data/fastvlm"),
            std::path::Path::new("../data/fastvlm"),
            std::path::Path::new("../../data/fastvlm"),
            std::path::Path::new("."),
        ];
        
        // 首先检查模型是否存在于任何可能的路径中
        let existing_dir = possible_paths.iter()
            .find(|path| {
                let exists = path.exists() && 
                    path.join("tokenizer.json").exists() &&
                    path.join("vision_encoder.onnx").exists() &&
                    path.join("embed_tokens.onnx").exists() &&
                    path.join("decoder_model_merged.onnx").exists();
                
                println!("🔍 检查路径 {:?}: {}", path, if exists { "✅ 找到所有模型" } else { "❌ 缺少模型" });
                exists
            })
            .copied();
            
        let data_dir = if let Some(dir) = existing_dir {
            println!("找到 FastVLM 数据目录: {:?}", dir);
            dir.to_path_buf()
        } else {
            println!("📥 FastVLM 模型未找到，尝试自动下载...");
            let download_dir = fastvlm::download::get_default_model_dir();
            
            let rt = tokio::runtime::Runtime::new().ok()?;
            match rt.block_on(fastvlm::download::download_fastvlm_models(&download_dir)) {
                Ok(_) => {
                    println!("✅ FastVLM 模型下载成功到: {:?}", download_dir);
                    download_dir
                },
                Err(e) => {
                    println!("❌ 下载 FastVLM 模型失败: {}", e);
                    return None;
                }
            }
        };
        
        let config = fastvlm::FastVLMConfig::default();
        match tokio::runtime::Runtime::new() {
            Ok(rt) => {
                match rt.block_on(fastvlm::FastVLM::new(&data_dir, config)) {
                    Ok(fastvlm) => {
                        println!("FastVLM 初始化成功");
                        Some(fastvlm)
                    },
                    Err(e) => {
                        println!("初始化 FastVLM 失败: {}", e);
                        None
                    }
                }
            },
            Err(e) => {
                println!("创建 tokio runtime 失败: {}", e);
                None
            }
        }
    }
    
    fn load_image(&self, image_path: &str) -> Result<DynamicImage> {
        println!("📸 加载图片: {}", image_path);
        let img = image::open(image_path)?;
        println!("✅ 图片加载成功: {}x{}", img.width(), img.height());
        Ok(img)
    }
    
    fn resize_image(&self, img: &DynamicImage, target_width: u32, target_height: u32) -> DynamicImage {
        img.resize(target_width, target_height, image::imageops::FilterType::Lanczos3)
    }
    
    fn image_to_rgba_bytes(&self, img: &DynamicImage) -> Vec<u8> {
        let rgba_img = img.to_rgba8();
        let mut bytes = Vec::with_capacity((rgba_img.width() * rgba_img.height() * 4) as usize);
        
        for pixel in rgba_img.pixels() {
            bytes.extend_from_slice(&[pixel[0], pixel[1], pixel[2], pixel[3]]);
        }
        
        bytes
    }
    
    fn analyze_image(&mut self, image_path: &str) -> Result<ImageAnalysisResult> {
        let start_time = Instant::now();
        
        // 加载图片
        let img = self.load_image(image_path)?;
        
        // 调整图片大小以提高处理速度
        let target_width = (img.width() as f32 * self.llm_resolution_scale) as u32;
        let target_height = (img.height() as f32 * self.llm_resolution_scale) as u32;
        let resized_img = self.resize_image(&img, target_width, target_height);
        
        println!("🔄 调整图片大小到: {}x{}", target_width, target_height);
        
        // 转换为 RGBA 字节
        let image_bytes = self.image_to_rgba_bytes(&resized_img);
        
        // 使用 FastVLM 分析
        if let Some(ref mut fastvlm) = self.fastvlm {
            println!("🤖 开始 AI 分析...");
            match fastvlm.analyze_frame_sync(image_bytes, target_width, target_height, Some(self.llm_prompt.clone())) {
                Ok(result) => {
                    let processing_time = start_time.elapsed();
                    println!("✅ 分析完成！耗时: {:.2}秒", processing_time.as_secs_f32());
                    
                    Ok(ImageAnalysisResult {
                        text: result.text,
                        processing_time,
                        timestamp: Instant::now(),
                    })
                },
                Err(e) => {
                    println!("❌ AI 分析失败: {}", e);
                    Err(anyhow::anyhow!("AI 分析失败: {}", e))
                }
            }
        } else {
            Err(anyhow::anyhow!("FastVLM 未初始化"))
        }
    }
    
    fn process_image_batch(&mut self, image_paths: &[String]) -> Result<()> {
        let batch_start_time = Instant::now();
        println!("🚀 开始批量处理 {} 张图片", image_paths.len());
        
        let mut stats = ProcessingStats::new();
        
        for (i, image_path) in image_paths.iter().enumerate() {
            println!("\n--- 处理第 {} 张图片: {} ---", i + 1, image_path);
            
            let image_start_time = Instant::now();
            match self.analyze_image(image_path) {
                Ok(result) => {
                    let image_processing_time = image_start_time.elapsed();
                    println!("📝 分析结果: {}", result.text);
                    println!("⏱️  单张图片处理时间: {:.2}秒", image_processing_time.as_secs_f32());
                    
                    stats.add_result(image_processing_time, true);
                },
                Err(e) => {
                    let image_processing_time = image_start_time.elapsed();
                    println!("❌ 处理失败: {} (耗时: {:.2}秒)", e, image_processing_time.as_secs_f32());
                    stats.add_result(image_processing_time, false);
                }
            }
        }
        
        let total_batch_time = batch_start_time.elapsed();
        stats.finalize();
        
        println!("\n✅ 批量处理完成！");
        println!("📊 总批次处理时间: {:.2}秒", total_batch_time.as_secs_f32());
        
        // 打印详细统计信息
        stats.print_summary();
        
        // 如果有初始化时间，也显示
        if let Some(init_time) = self.init_time {
            println!("🔧 初始化耗时: {:.2}秒", init_time.as_secs_f32());
            println!("⏱️  总耗时 (初始化 + 处理): {:.2}秒", (init_time + total_batch_time).as_secs_f32());
        }
        
        Ok(())
    }
}

fn main() -> Result<()> {
    env_logger::init();
    
    println!("🚀 启动 FastVLM - 图片 AI 分析工具");
    
    // 检查命令行参数
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 2 {
        println!("使用方法: {} <图片路径1> [图片路径2] [图片路径3] ...", args[0]);
        println!("示例: {} image1.jpg image2.png", args[0]);
        return Ok(());
    }
    
    // 初始化应用
    let mut app = FastVLMApp::new()?;
    
    // 获取图片路径列表
    let image_paths: Vec<String> = args[1..].to_vec();
    
    // 验证图片文件是否存在
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
        return Ok(());
    }
    
    // 处理图片
    app.process_image_batch(&valid_paths)?;
    
    Ok(())
}