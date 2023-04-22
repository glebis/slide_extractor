import cv2
import numpy as np
import sys
import os
import argparse
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def calculate_histogram(frame, num_bins=64):
    hist = np.zeros(num_bins * 3, dtype=np.float32)
    for channel in range(3):
        channel_hist = cv2.calcHist([frame], [channel], None, [num_bins], [0, 256])
        hist[channel * num_bins:(channel + 1) * num_bins] = channel_hist[:, 0]
    return hist

def histogram_difference(frame1, frame2, num_bins=64):
    hist1 = calculate_histogram(frame1, num_bins)
    hist2 = calculate_histogram(frame2, num_bins)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

def detect_scene_changes(video_path, threshold=0.5, num_bins=64):
    video = cv2.VideoCapture(video_path)

    ret, prev_frame = video.read()
    prev_hist = calculate_histogram(prev_frame, num_bins)

    scene_changes = []

    while True:
        ret, curr_frame = video.read()
        if not ret:
            break

        curr_hist = calculate_histogram(curr_frame, num_bins)
        diff = histogram_difference(prev_frame, curr_frame, num_bins)

        if diff > threshold:
            scene_changes.append(int(video.get(cv2.CAP_PROP_POS_FRAMES)))

        prev_frame = curr_frame
        prev_hist = curr_hist

    video.release()
    return scene_changes

def is_similar(frame1, frame2, threshold=0.9):
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    similarity = ssim(frame1_gray, frame2_gray)
    return similarity > threshold

def detect_slides(video_path, min_slide_duration=3, similarity_threshold=0.9, scene_threshold=0.5):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    min_slide_frames = int(min_slide_duration * fps)

    scene_changes = detect_scene_changes(video_path, threshold=scene_threshold)
    slides = []

    for idx, start_frame in enumerate(scene_changes):
        end_frame = frame_count if idx == len(scene_changes) - 1 else scene_changes[idx + 1]

        slide_start = start_frame
        slide_end = slide_start

        for frame_number in range(start_frame, min(start_frame + min_slide_frames, end_frame)):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = video.read()
            if not ret:
                break

            if frame_number == start_frame:
                prev_frame = frame
            else:
                if is_similar(prev_frame, frame, threshold=similarity_threshold):
                    slide_end = frame_number
                else:
                    break

        if slide_end - slide_start >= min_slide_frames - 1:
            slides.append({
                "start_frame": slide_start,
                "end_frame": slide_end,
                "duration": (slide_end - slide_start) / fps
            })

    video.release()
    return slides

def save_slides_as_images(slides, video_path, output_folder):
    video = cv2.VideoCapture(video_path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, slide in enumerate(slides):
        video.set(cv2.CAP_PROP_POS_FRAMES, slide["start_frame"])
        ret, frame = video.read()
        if not ret:
            continue

        img_path = os.path.join(output_folder, f"slide_{i}.png")
        cv2.imwrite(img_path, frame)

    video.release()
    return len(slides)

def save_slides_as_pdf(slides, video_path, output_path):
    if not slides:
        print("No slides detected. PDF will not be created.")
        return 0, 0

    video = cv2.VideoCapture(video_path)
    c = canvas.Canvas(output_path, pagesize=A4)

    for i, slide in enumerate(slides):
        video.set(cv2.CAP_PROP_POS_FRAMES, slide["start_frame"])
        ret, frame = video.read()
        if not ret:
            continue

        img_path = f"temp_slide_{i}.png"
        cv2.imwrite(img_path, frame)

        img = Image.open(img_path)
        img_width, img_height = img.size
        pdf_width, pdf_height = A4
        scale = min(pdf_width / img_width, pdf_height / img_height)

        img_width *= scale
        img_height *= scale
        x = (pdf_width - img_width) / 2
        y = (pdf_height - img_height) / 2

        c.drawImage(img_path, x, y, img_width, img_height)
        c.showPage()
        os.remove(img_path)

    c.save()
    return len(slides), os.path.getsize(output_path)

def main():
    parser = argparse.ArgumentParser(description="Detect presentation slides in a video")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--min_slide_duration", type=int, default=3, help="Minimum slide duration in seconds (default: 3)")
    parser.add_argument("--similarity_threshold", type=float, default=0.9, help="Similarity threshold for slide detection (default: 0.9)")
    parser.add_argument("--scene_threshold", type=float, default=0.5, help="Scene change threshold (default: 0.5)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run, only output detected slides")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--output_type", choices=["pdf", "images"], default="pdf", help="Output type: 'pdf' or 'images' (default: pdf)")

    args = parser.parse_args()

    if not args.output:
        args.output = os.path.splitext(args.video_path)[0] + ".pdf" if args.output_type == "pdf" else os.path.splitext(args.video_path)[0] + "_slides"

    slides = detect_slides(args.video_path, min_slide_duration=args.min_slide_duration, similarity_threshold=args.similarity_threshold, scene_threshold=args.scene_threshold)

    if args.dry_run:
        for i, slide in enumerate(slides):
            print(f"Slide {i}: start_frame={slide['start_frame']}, end_frame={slide['end_frame']}, duration={slide['duration']}s")
    else:
        if args.output_type == "pdf":
            num_slides, file_size = save_slides_as_pdf(slides, args.video_path, args.output)
            print(f"PDF exported with {num_slides} slides and a file size of {file_size} bytes")
        else:
            num_slides = save_slides_as_images(slides, args.video_path, args.output)
            print(f"{num_slides} images exported")

if __name__ == "__main__":
    main()
