import argparse
import os
import random

def process(bg_image_path, fg_images_paths, args, original_res=(480, 640)):
    
    bg_name = os.path.basename(bg_image_path).split('.')[0]
    bg_folder = os.path.join(args.output_dir, bg_name)
    os.makedirs(bg_folder, exist_ok=True)
    
    height, width = original_res
    tmax = 1.0
    median_filter_size_bg = 15.0
    gaussian_blur_sigma_bg= 1.0
    median_filter_size_fg = 11.0
    gaussian_blur_sigma_fg= 1.0
    
    for fg_image in fg_images_paths:
        fg_name = os.path.basename(fg_image).split('.')[0]
        scene_filename = os.path.join(bg_folder, f"{bg_name}_{fg_name}.scene")
        
        theta0_bg, theta1_bg = round(random.uniform(-5, 5), 2), round(random.uniform(-5, 5), 2)
        x0_bg, x1_bg = round(random.uniform(-0.2, 0.2), 2), round(random.uniform(-0.2, 0.2), 2)
        y0_bg, y1_bg = x0_bg, x1_bg
        sx0_bg, sx1_bg = round(random.uniform(0.9,1.1),2), round(random.uniform(0.9,1.1),2) 
        sy0_bg, sy1_bg = sx0_bg, sx1_bg
        
        # Ensure foreground moves faster than the background
        theta0_fg, theta1_fg = round(random.uniform(-10, 10), 1), round(theta1_bg + random.uniform(2, 5), 1)
        x0_fg, x1_fg = round(random.uniform(-1.5, 1.5), 2), round(x1_bg + random.uniform(-1.5, 1.5), 2)
        y0_fg, y1_fg = x0_fg, x1_fg
        sx0_fg, sx1_fg = round(random.uniform(0.5,1.2),2), round(random.uniform(0.5,1.2),2) 
        sy0_fg, sy1_fg = sx0_bg, sx1_bg
        
        scene_content = f"{width} {height} {tmax}\n"
        scene_content += f"{bg_image_path} {median_filter_size_bg} {gaussian_blur_sigma_bg} "
        scene_content += f"{theta0_bg} {theta1_bg} {x0_bg} {x1_bg} {y0_bg} {y1_bg} {sx0_bg} {sx1_bg} {sy0_bg} {sy1_bg}\n"
        scene_content += f"{fg_image} {median_filter_size_fg} {gaussian_blur_sigma_fg} "
        scene_content += f"{theta0_fg} {theta1_fg} {x0_fg} {x1_fg} {y0_fg} {y1_fg} {sx0_fg} {sx1_fg} {sy0_fg} {sy1_fg}\n"
        
        with open(scene_filename, 'w') as f:
            f.write(scene_content)
        print(f"Generated {scene_filename}")


if __name__ == "__main__":
    """
    Tool for generating Scene files
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--bg_path", default="/home/yousef/Documents/train/images/background/")
    parser.add_argument("--fg_path", default="/home/yousef/Documents/train/images/foreground/")
    parser.add_argument("--output_dir", default="/home/yousef/Documents/train/scenes/")
    args = parser.parse_args()

    # Get background and foreground files
    bg_images_paths = [os.path.join(args.bg_path, file) for file in os.listdir(args.bg_path) if file.endswith((".png", ".jpg"))]
    fg_images_paths = [os.path.join(args.fg_path, file) for file in os.listdir(args.fg_path) if file.endswith((".png", ".jpg"))]
    
    # make sure output directory exists
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # process files
    for bg_image_path in bg_images_paths:
        process(bg_image_path, fg_images_paths, args)

