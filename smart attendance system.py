import cv2
import numpy as np
import json
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

students_file = 'students.json'
attendance_file = 'attendance.json'

class SmartAttendanceApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Smart Attendance System")

        self.init_files()
        self.create_widgets()

        self.recognition_stopped = True  # Flag to track recognition state

    def init_files(self):
        if not os.path.exists(students_file):
            with open(students_file, 'w') as f:
                json.dump([], f)
        if not os.path.exists(attendance_file):
            with open(attendance_file, 'w') as f:
                json.dump([], f)

    def create_widgets(self):
        self.tabControl = ttk.Notebook(self)
        self.tabControl.pack(expand=1, fill="both")

        self.tab1 = ttk.Frame(self.tabControl)
        self.tabControl.add(self.tab1, text="Attendance Records")

        self.tab2 = ttk.Frame(self.tabControl)
        self.tabControl.add(self.tab2, text="Add Student")

        self.tab3 = ttk.Frame(self.tabControl)
        self.tabControl.add(self.tab3, text="Start Recognition")

        self.create_attendance_records_tab()
        self.create_add_student_tab()
        self.create_start_recognition_tab()

    def create_attendance_records_tab(self):
        self.records_frame = ttk.Frame(self.tab1)
        self.records_frame.pack(fill="both", expand=True)

        self.records_table = ttk.Treeview(self.records_frame, columns=("ID", "Name", "Date", "Check In", "Check Out"))
        self.records_table.heading("#0", text="ID")
        self.records_table.heading("Name", text="Name")
        self.records_table.heading("Date", text="Date")
        self.records_table.heading("Check In", text="Check In")
        self.records_table.heading("Check Out", text="Check Out")
        self.records_table.pack(fill="both", expand=True)

        self.load_attendance_records()

    def create_add_student_tab(self):
        self.add_student_frame = ttk.Frame(self.tab2)
        self.add_student_frame.pack(fill="both", expand=True)

        self.name_label = ttk.Label(self.add_student_frame, text="Name:")
        self.name_label.grid(row=0, column=0, padx=5, pady=5)

        self.name_entry = ttk.Entry(self.add_student_frame)
        self.name_entry.grid(row=0, column=1, padx=5, pady=5)

        self.add_button = ttk.Button(self.add_student_frame, text="Add", command=self.add_student)
        self.add_button.grid(row=1, columnspan=2, padx=5, pady=5)

    def create_start_recognition_tab(self):
        self.recognition_frame = ttk.Frame(self.tab3)
        self.recognition_frame.pack(fill="both", expand=True)

        self.video_label = tk.Label(self.recognition_frame)
        self.video_label.pack(padx=10, pady=10)

        self.start_button = ttk.Button(self.recognition_frame, text="Start Recognition", command=self.start_recognition)
        self.start_button.pack(padx=10, pady=5)

        self.stop_button = ttk.Button(self.recognition_frame, text="Stop Recognition", command=self.stop_recognition, state="disabled")
        self.stop_button.pack(padx=10, pady=5)

    def load_attendance_records(self):
        for item in self.records_table.get_children():
            self.records_table.delete(item)
        attendance = self.load_from_file(attendance_file)
        students = self.load_from_file(students_file)
        for record in attendance:
            student_name = next(student['name'] for student in students if student['id'] == record['student_id'])
            self.records_table.insert("", "end", text=str(record['student_id']),
                                      values=(record['student_id'], student_name, record['date'], record.get('check_in', ''), record.get('check_out', '')))

    def save_to_file(self, file, data):
        with open(file, 'w') as f:
            json.dump(data, f, indent=4)

    def load_from_file(self, file):
        with open(file, 'r') as f:
            return json.load(f)

    def capture_image(self, student_name):
        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()
        if ret:
            img_name = f"{student_name}.png"
            cv2.imwrite(img_name, frame)
            print(f"{img_name} written!")
        cam.release()
        cv2.destroyAllWindows()

    def encode_faces(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        students = self.load_from_file(students_file)
        for student in students:
            img_path = f"{student['name']}.png"
            if not os.path.exists(img_path):
                print(f"Error: Image {img_path} not found")
                continue
            image = cv2.imread(img_path)
            if image is None:
                print(f"Error: Failed to read image {img_path}")
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                print(f"Error: No faces detected in image {img_path}")
                continue
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (100, 100))
                face_data = face.flatten().tolist()
                student['face_data'] = face_data
        self.save_to_file(students_file, students)

    def recognize_faces(self):
        self.recognition_stopped = False  # Set recognition flag
        self.stop_button.config(state="normal")  # Enable stop button
        self.start_button.config(state="disabled")  # Disable start button

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        video_capture = cv2.VideoCapture(0)
        students = self.load_from_file(students_file)
        attendance = self.load_from_file(attendance_file)

        while not self.recognition_stopped:  # Loop until recognition stopped
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Failed to capture frame from camera")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                print("Error: No faces detected in frame")
                continue
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (100, 100))
                face_data = face.flatten()
                name = "Unknown"
                min_dist = float('inf')
                for student in students:
                    if 'face_data' not in student:
                        continue  # Skip if face data is missing
                    db_face_data = np.array(student['face_data'], dtype=np.uint8)
                    if db_face_data.size == 0:
                        continue  # Skip if face data is empty
                    dist = np.linalg.norm(face_data - db_face_data)
                    if dist < min_dist:
                        min_dist = dist
                        name = student['name']
                        student_id = student['id']
                if min_dist < 5000:
                    today = datetime.now().strftime("%Y-%m-%d")
                    now_time = datetime.now().strftime("%H:%M:%S")
                    attendance_record = {
                        "student_id": student_id,
                        "date": today,
                        "check_in": now_time
                    }
                    for record in attendance:
                        if record['student_id'] == student_id and record['date'] == today:
                            if 'check_out' not in record:
                                record['check_out'] = now_time
                                self.save_to_file(attendance_file, attendance)
                                break
                    else:
                        attendance.append(attendance_record)
                        self.save_to_file(attendance_file, attendance)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            self.video_label.img = img
            self.video_label.configure(image=img)
            self.video_label.update()

        # Recognition stopped, enable start button and disable stop button
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

        video_capture.release()
        cv2.destroyAllWindows()

    def stop_recognition(self):
        self.recognition_stopped = True  # Set recognition flag

    def add_student(self):
        name = self.name_entry.get()
    def add_student(self):
         name = self.name_entry.get()
         if name:
            students = self.load_from_file(students_file)
            student_id = len(students) + 1
            students.append({"id": student_id, "name": name})
            self.save_to_file(students_file, students)

            self.capture_image(name)
            self.encode_faces()

            self.name_entry.delete(0, tk.END)
            self.load_attendance_records()

            print(f"Student {name} added successfully with ID {student_id}")

         else:
            print("Error: Student name cannot be empty")

if __name__ == "__main__":
    app = SmartAttendanceApp()
    app.mainloop()

       
