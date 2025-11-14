import cv2
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from threading import Thread
import numpy as np
import subprocess
import tempfile
import time

class FaceVideoProcessor:
    def __init__(self, master):
        self.master = master
        self.master.title("Face Video Processor - by Yofo")
        
        # Configuration
        self.model_path = "D:/programs/facedetect/res10_300x300_ssd_iter_140000.caffemodel"
        self.config_path = "D:/programs/facedetect/deploy.prototxt"
        self.cascade_path = "D:/programs/facedetect/haarcascade_frontalface_default.xml"
        self.ffmpeg_path = "C:/FFmpeg/bin/ffmpeg.exe"  # Full path to FFmpeg
        self.input_path = ""
        self.output_path = ""
        self.processing = False
        self.frame_count = 0
        self.kept_frames = 0
        self.confidence_threshold = 0.5
        self.detection_method = "SSD"  # Default to SSD
        self.has_audio = False  # Flag to track if video has audio
        
        # Progress tracking variables
        self.start_time = None
        self.processing_speed = 0  # frames per second
        self.last_update_time = None
        self.frames_since_last_update = 0

        # Feature flags
        self.frame_output_format = tk.StringVar(value="none")  # Changed default to "none"
        self.include_audio = tk.BooleanVar(value=True)

        # Create GUI elements
        self.create_widgets()
        self.load_model()
        self.check_ffmpeg()

    def check_ffmpeg(self):
        try:
            # Test FFmpeg installation
            test_cmd = [self.ffmpeg_path, '-version']
            process = subprocess.Popen(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise Exception("FFmpeg test failed")
            self.update_progress("✅ FFmpeg is properly installed")
        except Exception as e:
            messagebox.showerror("Error", f"FFmpeg not found or not working at: {self.ffmpeg_path}\nPlease make sure FFmpeg is installed correctly.")
            self.master.destroy()
            return

    def run_ffmpeg_command(self, cmd, error_msg):
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW  # Prevent console window
            )
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd, stderr.decode())
            return True
        except subprocess.CalledProcessError as e:
            self.update_progress(f"❌ {error_msg}: {e.stderr}")
            return False
        except Exception as e:
            self.update_progress(f"❌ {error_msg}: {str(e)}")
            return False

    def create_widgets(self):
        # Input Section
        input_frame = ttk.LabelFrame(self.master, text="Input Settings")
        input_frame.pack(padx=10, pady=5, fill=tk.X)

        ttk.Button(input_frame, text="Browse Video File", 
                 command=self.browse_input).grid(row=0, column=0, padx=5, pady=5)
        self.input_label = ttk.Label(input_frame, text="No file selected")
        self.input_label.grid(row=0, column=1, padx=5, sticky=tk.W)

        # Output Section
        output_frame = ttk.LabelFrame(self.master, text="Output Settings")
        output_frame.pack(padx=10, pady=5, fill=tk.X)

        ttk.Button(output_frame, text="Choose Save Location", 
                 command=self.browse_output).grid(row=0, column=0, padx=5, pady=5)
        self.output_label = ttk.Label(output_frame, text="Default: input folder")
        self.output_label.grid(row=0, column=1, padx=5, sticky=tk.W)

        # Detection Settings Section
        detection_frame = ttk.LabelFrame(self.master, text="Detection Settings")
        detection_frame.pack(padx=10, pady=5, fill=tk.X)

        # Detection Method Selection
        ttk.Label(detection_frame, text="Detection Method:").grid(row=0, column=0, padx=5, pady=5)
        self.method_var = tk.StringVar(value="SSD")
        method_combo = ttk.Combobox(detection_frame, textvariable=self.method_var, 
                                  values=["SSD", "Haar Cascade"], state="readonly")
        method_combo.grid(row=0, column=1, padx=5, pady=5)
        method_combo.bind('<<ComboboxSelected>>', self.on_method_change)

        # Confidence Threshold Slider
        ttk.Label(detection_frame, text="Confidence Threshold:").grid(row=1, column=0, padx=5, pady=5)
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = ttk.Scale(detection_frame, from_=0.1, to=1.0, 
                                   variable=self.confidence_var, orient=tk.HORIZONTAL,
                                   command=self.on_confidence_change)
        confidence_scale.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        self.confidence_label = ttk.Label(detection_frame, text="0.5")
        self.confidence_label.grid(row=1, column=2, padx=5, pady=5)

        # Feature Options Section
        features_frame = ttk.LabelFrame(self.master, text="Feature Options")
        features_frame.pack(padx=10, pady=5, fill=tk.X)

        # Frame Output Format Selection
        self.frame_format_label = ttk.Label(features_frame, text="Frame Output Format:")
        self.frame_format_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.frame_format_combo = ttk.Combobox(
            features_frame,
            textvariable=self.frame_output_format,
            values=["none", "images", "video", "both"],
            state="readonly"
        )
        self.frame_format_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        self.frame_format_combo.bind('<<ComboboxSelected>>', self.on_frame_format_change)

        # Include Audio Checkbox
        self.include_audio_cb = ttk.Checkbutton(
            features_frame,
            text="Include audio in output",
            variable=self.include_audio
        )
        self.include_audio_cb.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)

        # Control Section
        control_frame = ttk.Frame(self.master)
        control_frame.pack(padx=10, pady=10, fill=tk.X)

        self.start_btn = ttk.Button(control_frame, text="Start Processing", 
                                  command=self.start_processing)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(control_frame, text="Stop", 
                                  state=tk.DISABLED, command=self.stop_processing)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.open_output_btn = ttk.Button(control_frame, text="Open Output Video", 
                                        command=self.open_output_video, state=tk.DISABLED)
        self.open_output_btn.pack(side=tk.LEFT, padx=5)

        # Progress Section
        progress_frame = ttk.LabelFrame(self.master, text="Progress")
        progress_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        # Progress Bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, 
                                          variable=self.progress_var,
                                          maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Progress Percentage Label
        self.progress_label = ttk.Label(progress_frame, text="0%")
        self.progress_label.pack(pady=2)
        
        # Status Label
        self.status_label = ttk.Label(progress_frame, 
                                    text="Ready to process")
        self.status_label.pack(pady=2)

        # Progress Text
        self.progress_text = scrolledtext.ScrolledText(progress_frame, height=8)
        self.progress_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def on_method_change(self, event=None):
        self.detection_method = self.method_var.get()
        self.load_model()
        self.update_progress(f"Switched to {self.detection_method} detection method")
        # Update output filename if input file is selected
        if self.input_path:
            original_filename = os.path.splitext(os.path.basename(self.input_path))[0]
            default_output = os.path.join(
                os.path.dirname(self.input_path),
                f"FaceOnly_{original_filename}_threshold_{self.confidence_threshold:.2f}_{self.detection_method}.mp4"
            )
            self.output_label.config(text=default_output)

    def on_confidence_change(self, event=None):
        self.confidence_threshold = self.confidence_var.get()
        self.confidence_label.config(text=f"{self.confidence_threshold:.2f}")
        # Update output filename if input file is selected
        if self.input_path:
            original_filename = os.path.splitext(os.path.basename(self.input_path))[0]
            default_output = os.path.join(
                os.path.dirname(self.input_path),
                f"FaceOnly_{original_filename}_threshold_{self.confidence_threshold:.2f}_{self.detection_method}.mp4"
            )
            self.output_label.config(text=default_output)

    def load_model(self):
        try:
            if self.detection_method == "SSD":
                self.face_detector = cv2.dnn.readNetFromCaffe(self.config_path, self.model_path)
                if self.face_detector.empty():
                    raise Exception("Failed to load SSD model")
            else:  # Haar Cascade
                self.face_detector = cv2.CascadeClassifier(self.cascade_path)
                if self.face_detector.empty():
                    raise Exception("Failed to load Haar Cascade model")
        except Exception as e:
            messagebox.showerror("Error", f"Couldn't load face detection model!\nError: {str(e)}")
            self.master.destroy()

    def check_video_audio(self):
        """Check if the input video has audio"""
        try:
            cmd = [
                self.ffmpeg_path, '-i', self.input_path
            ]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            _, stderr = process.communicate()
            stderr_str = stderr.decode()
            self.has_audio = "Stream #0:1" in stderr_str or "Stream #0:0: Audio" in stderr_str
            return True
        except Exception as e:
            self.update_progress(f"Warning: Could not check audio: {str(e)}")
            return False

    def open_output_video(self):
        """Open the output video file"""
        if not self.output_path or not os.path.exists(self.output_path):
            messagebox.showwarning("Warning", "No output video file found!")
            return
        
        try:
            os.startfile(self.output_path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open video: {str(e)}")

    def browse_input(self):
        self.input_path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mov"), ("All Files", "*.*")]
        )
        if self.input_path:
            self.input_label.config(text=self.input_path)
            # Check if video has audio
            self.check_video_audio()
            # Get original filename without extension
            original_filename = os.path.splitext(os.path.basename(self.input_path))[0]
            # Create default output path with new format
            default_output = os.path.join(
                os.path.dirname(self.input_path),
                f"FaceOnly_{original_filename}_threshold_{self.confidence_threshold:.2f}_{self.detection_method}.mp4"
            )
            self.output_label.config(text=default_output)
            self.output_path = default_output

    def browse_output(self):
        # Get original filename without extension
        original_filename = os.path.splitext(os.path.basename(self.input_path))[0]
        # Create suggested filename with new format
        suggested_filename = f"FaceOnly_{original_filename}_threshold_{self.confidence_threshold:.2f}_{self.detection_method}.mp4"
        
        self.output_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            initialfile=suggested_filename,
            filetypes=[("MP4 Files", "*.mp4"), ("All Files", "*.*")]
        )
        if self.output_path:
            self.output_label.config(text=self.output_path)

    def start_processing(self):
        if not self.input_path:
            messagebox.showwarning("Warning", "Please select an input video file!")
            return

        self.processing = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.frame_count = 0
        self.kept_frames = 0
        
        self.start_time = time.time()
        self.processing_speed = 0
        self.last_update_time = self.start_time
        self.frames_since_last_update = 0
        
        Thread(target=self.process_video, daemon=True).start()

    def stop_processing(self):
        self.processing = False
        self.update_progress("Processing stopped by user!")

    def update_progress(self, message):
        self.progress_text.insert(tk.END, message + "\n")
        self.progress_text.see(tk.END)
        self.master.update_idletasks()

    def on_frame_format_change(self, event=None):
        """Handle frame output format change"""
        if self.input_path:
            format_type = self.frame_output_format.get()
            if format_type != "none":
                frames_dir = os.path.join(os.path.dirname(self.input_path), 
                                        f"frames_{os.path.splitext(os.path.basename(self.input_path))[0]}_{self.detection_method}_{time.strftime('%Y%m%d_%H%M%S')}")
                if not os.path.exists(frames_dir):
                    os.makedirs(frames_dir)
                
                if format_type in ["images", "both"]:
                    self.update_progress(f"📸 Frames will be extracted as images to: {frames_dir}")
                if format_type in ["video", "both"]:
                    self.update_progress(f"🎥 Frames will be saved as video in: {frames_dir}")

    def process_video(self):
        try:
            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open video file!")
                return

            # Get total frame count for percentage calculation
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Define video codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # Initialize progress tracking
            self.start_time = time.time()
            self.last_update_time = self.start_time
            self.frames_since_last_update = 0
            self.processing_speed = 0

            # Get original filename without extension
            original_filename = os.path.splitext(os.path.basename(self.input_path))[0]
            
            # Create temporary files with shorter names
            temp_dir = tempfile.gettempdir()
            temp_video = os.path.join(temp_dir, "temp_vid.mp4")
            temp_audio = os.path.join(temp_dir, "temp_aud.mp3")
            temp_trimmed_audio = os.path.join(temp_dir, "temp_trimmed_aud.mp3")
            temp_final = os.path.join(temp_dir, "temp_final.mp4")
            
            # Create final output path with new format
            output_dir = os.path.dirname(self.input_path)
            output_name = f"FaceOnly_{original_filename[:20]}_t{self.confidence_threshold:.2f}_{self.detection_method[:4]}.mp4"
            final_output = os.path.join(output_dir, output_name)
            self.output_path = final_output

            # Create frames directory if needed
            frames_dir = None
            format_type = self.frame_output_format.get()
            if format_type != "none":
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                frames_dir = os.path.join(output_dir, f"frames_{original_filename}_{self.detection_method}_{timestamp}")
                if not os.path.exists(frames_dir):
                    os.makedirs(frames_dir)
                
                if format_type in ["images", "both"]:
                    self.update_progress(f"📸 Frames will be extracted as images to: {frames_dir}")
                if format_type in ["video", "both"]:
                    self.update_progress(f"🎥 Frames will be saved as video in: {frames_dir}")

                # Initialize video writer for frames if needed
                frames_video_writer = None
                if format_type in ["video", "both"]:
                    frames_video_path = os.path.join(frames_dir, f"frames_video_{timestamp}.mp4")
                    frames_video_writer = cv2.VideoWriter(frames_video_path, fourcc, fps, (width, height))

            out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

            self.update_progress("⏳ Processing video...")
            self.update_progress(f"Total frames to process: {total_frames}")

            # List to store frame timestamps
            frame_timestamps = []
            current_time = 0
            frame_duration = 1.0 / fps
            last_percentage = -1  # Track last reported percentage
            frame_counter = 0  # Counter for extracted frames

            while self.processing:
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1
                self.frames_since_last_update += 1
                
                # Update processing speed every second
                current_time_elapsed = time.time() - self.last_update_time
                if current_time_elapsed >= 1.0:
                    self.processing_speed = self.frames_since_last_update / current_time_elapsed
                    self.frames_since_last_update = 0
                    self.last_update_time = time.time()
                
                if self.detection_method == "SSD":
                    # SSD Detection
                    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                    self.face_detector.setInput(blob)
                    detections = self.face_detector.forward()
                    
                    # Process detections
                    face_found = False
                    max_confidence = 0.0
                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > self.confidence_threshold:
                            face_found = True
                            max_confidence = max(max_confidence, confidence)
                else:
                    # Haar Cascade Detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
                    face_found = len(faces) > 0
                    max_confidence = 1.0 if face_found else 0.0  # Haar doesn't provide confidence, so use 1.0 if face found

                if face_found:
                    out.write(frame)
                    self.kept_frames += 1
                    frame_timestamps.append((current_time, current_time + frame_duration))
                    
                    # Extract frame if enabled
                    if format_type != "none":
                        if format_type in ["images", "both"]:
                            frame_path = os.path.join(frames_dir, f"frame_{frame_counter:06d}_conf_{max_confidence:.3f}.jpg")
                            cv2.imwrite(frame_path, frame)
                        if format_type in ["video", "both"]:
                            frames_video_writer.write(frame)
                        frame_counter += 1
                
                current_time += frame_duration

                # Calculate progress and time remaining
                current_percentage = (self.frame_count / total_frames) * 100
                frames_remaining = total_frames - self.frame_count
                if self.processing_speed > 0:
                    time_remaining = frames_remaining / self.processing_speed
                    time_remaining_str = self.format_time(time_remaining)
                else:
                    time_remaining_str = "Calculating..."

                # Update progress bar and labels
                self.progress_var.set(current_percentage)
                self.progress_label.config(text=f"{current_percentage:.1f}%")
                self.status_label.config(
                    text=f"Frame {self.frame_count}/{total_frames} - {time_remaining_str} remaining - {self.processing_speed:.1f} fps"
                )

                # Update progress text every 5%
                if int(current_percentage) != last_percentage and int(current_percentage) % 5 == 0:
                    kept_percentage = (self.kept_frames / self.frame_count) * 100
                    self.update_progress(
                        f"Progress: {current_percentage:.1f}% complete ({kept_percentage:.1f}% frames kept)"
                    )
                    if format_type != "none":
                        self.update_progress(f"📸 Extracted {frame_counter} frames so far")
                    last_percentage = int(current_percentage)

            cap.release()
            out.release()

            if not self.processing:
                self.update_progress("Processing stopped by user!")
                return

            if not frame_timestamps:
                self.update_progress("❌ No frames with faces were detected!")
                return

            self.update_progress("🎥 Video processing complete!")

            # Process audio if enabled and available
            if self.has_audio and self.include_audio.get():
                self.update_progress("🎵 Processing audio segments...")
                
                # Create a file list for concatenation
                file_list = os.path.join(temp_dir, "file_list.txt")
                with open(file_list, 'w') as f:
                    for i, (start, end) in enumerate(frame_timestamps):
                        segment_file = os.path.join(temp_dir, f'segment_{i:03d}.mp3')
                        
                        # Extract this segment using a simpler approach
                        segment_cmd = [
                            self.ffmpeg_path, '-y',
                            '-ss', str(start),
                            '-i', temp_audio,
                            '-t', str(end - start),
                            '-acodec', 'libmp3lame',
                            '-q:a', '2',
                            segment_file
                        ]
                        
                        if not self.run_ffmpeg_command(segment_cmd, f"Failed to process segment {start:.2f}-{end:.2f}"):
                            continue
                        
                        if os.path.exists(segment_file) and os.path.getsize(segment_file) > 0:
                            f.write(f"file '{segment_file}'\n")
                            self.update_progress(f"✅ Processed segment {start:.2f}-{end:.2f}")

                # Check if we have any valid segments
                if os.path.getsize(file_list) == 0:
                    self.update_progress("⚠️ No valid audio segments found, proceeding without audio")
                    combine_cmd = [
                        self.ffmpeg_path, '-y',
                        '-i', temp_video,
                        '-c:v', 'copy',
                        temp_final
                    ]
                else:
                    # Concatenate all segments
                    concat_cmd = [
                        self.ffmpeg_path, '-y',
                        '-f', 'concat',
                        '-safe', '0',
                        '-i', file_list,
                        '-acodec', 'libmp3lame',
                        '-q:a', '2',
                        temp_trimmed_audio
                    ]
                    
                    if not self.run_ffmpeg_command(concat_cmd, "Failed to concatenate audio segments"):
                        self.update_progress("⚠️ Failed to concatenate audio, proceeding without audio")
                        combine_cmd = [
                            self.ffmpeg_path, '-y',
                            '-i', temp_video,
                            '-c:v', 'copy',
                            temp_final
                        ]
                    else:
                        self.update_progress("🔄 Combining video with trimmed audio...")
                        # Combine processed video with trimmed audio
                        combine_cmd = [
                            self.ffmpeg_path, '-y',
                            '-i', temp_video,
                            '-i', temp_trimmed_audio,
                            '-c:v', 'copy',
                            '-c:a', 'aac',
                            '-b:a', '192k',
                            '-map', '0:v:0',
                            '-map', '1:a:0',
                            '-shortest',
                            temp_final
                        ]
            else:
                # Just copy the video without audio
                combine_cmd = [
                    self.ffmpeg_path, '-y',
                    '-i', temp_video,
                    '-c:v', 'copy',
                    temp_final
                ]
            
            if not self.run_ffmpeg_command(combine_cmd, "Failed to create final video"):
                return

            # Move the final file to the desired output location
            if os.path.exists(final_output):
                os.remove(final_output)
            os.rename(temp_final, final_output)

            # Clean up temporary files
            temp_files = [temp_video]
            if self.has_audio and self.include_audio.get():
                temp_files.extend([temp_audio, temp_trimmed_audio, file_list])
                temp_files.extend(temp_audio_chunks)  # Add chunk files to cleanup
            
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass

            self.update_progress(f"✅ Processing complete! Saved to: {final_output}")
            if format_type != "none":
                self.update_progress(f"📸 Extracted {frame_counter} frames to: {frames_dir}")
            self.update_progress(f"📊 Statistics:")
            self.update_progress(f"   - Total frames processed: {self.frame_count}")
            self.update_progress(f"   - Frames kept: {self.kept_frames}")
            self.update_progress(f"   - Frames removed: {self.frame_count - self.kept_frames}")
            self.update_progress(f"   - Kept frames percentage: {(self.kept_frames/self.frame_count)*100:.1f}%")
            
            # Enable the Open Output button
            self.open_output_btn.config(state=tk.NORMAL)

            # Clean up video writer for frames if it was created
            if format_type != "none" and frames_video_writer is not None:
                frames_video_writer.release()
                if format_type in ["video", "both"]:
                    self.update_progress(f"🎥 Frames video saved to: {frames_video_path}")

        except Exception as e:
            self.update_progress(f"❌ Unexpected error: {str(e)}")
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")
        finally:
            self.processing = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            # Ensure all temporary files are cleaned up
            try:
                for temp_file in [temp_video, temp_audio, temp_trimmed_audio, temp_final]:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            except:
                pass

    def format_time(self, seconds):
        """Format seconds into a human-readable time string"""
        if seconds < 60:
            return f"{seconds:.0f} sec"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.0f} min {seconds % 60:.0f} sec"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.0f} hr {minutes:.0f} min"

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceVideoProcessor(root)
    root.geometry("600x400")
    root.mainloop()
