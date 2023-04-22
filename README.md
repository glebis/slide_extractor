# Slide Extractor

Slide Extractor is a Python script that detects presentation slides in a video and exports them as a PDF or a folder of PNG images.

## Requirements

- Python 3.6 or later
- OpenCV
- Pillow
- ReportLab

## Installation

1. Install the required libraries:

```bash 
pip install opencv-python-headless Pillow reportlab
```

2. Save the slide_extractor.py script to your project directory.



## Usage

Run the script using the following command:

```python slide_extractor.py --video_path /path/to/your/video.mp4```


Additional command-line arguments:

* --min_slide_duration: Minimum duration of a slide in seconds (default: 3)
* --similarity_threshold: Similarity threshold for comparing frames (default: 0.95)
* --scene_threshold: Threshold for detecting scene changes (default: 30)
* --dry_run: Perform a dry run and print detected slides without exporting them
* --output_format: Choose the output format: 'pdf' or 'images' (default: 'pdf')

## Authors
- Gleb Kalinin
- OpenAI ChatGPT


## License

This project is licensed under the MIT License - see the LICENSE file for details.