import cv2
import numpy as np
import json
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

students_file = 'students.json'
attendance_file = 'attendance.json'
staff_file = 'staff.json'

class SmartAttendanceApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Smart Attendance System")

        self.init_files()
        self.create_widgets()

        self.recognition_stopped = True
        self.load_face_recognition_model()

    def init_files(self):
        for file in [students_file, attendance_file, staff_file]:
            if not os.path.exists(file):
                with open(file, 'w') as f:
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

        self.tab4 = ttk.Frame(self.tabControl)
        self.tabControl.add(self.tab4, text="Staff Management")

        self.create_attendance_records_tab()
        self.create_add_student_tab()
        self.create_start_recognition_tab()
        self.create_staff_management_tab()

    def create_attendance_records_tab(self):
        self.records_frame = ttk.Frame(self.tab1)
        self.records_frame.pack(fill="both", expand=True)

        self.records_table = ttk.Treeview(self.records_frame, columns=("ID", "Name", "Date", "Check In", "Check Out", "Status"))
        self.records_table.heading("#0", text="ID")
        self.records_table.heading("Name", text="Name")
        self.records_table.heading("Date", text="Date")
        self.records_table.heading("Check In", text="Check In")
        self.records_table.heading("Check Out", text="Check Out")
        self.records_table.heading("Status", text="Status")
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

    def create_staff_management_tab(self):
        self.staff_frame = ttk.Frame(self.tab4)
        self.staff_frame.pack(fill="both", expand=True)

        self.staff_id_label = ttk.Label(self.staff_frame, text="Staff ID:")
        self.staff_id_label.grid(row=0, column=0, padx=5, pady=5)

        self.staff_id_entry = ttk.Entry(self.staff_frame)
        self.staff_id_entry.grid(row=0, column=1, padx=5, pady=5)

        self.staff_name_label = ttk.Label(self.staff_frame, text="Staff Name:")
        self.staff_name_label.grid(row=1, column=0, padx=5, pady=5)

        self.staff_name_entry = ttk.Entry(self.staff_frame)
        self.staff_name_entry.grid(row=1, column=1, padx=5, pady=5)

        self.add_staff_button = ttk.Button(self.staff_frame, text="Add Staff", command=self.add_staff)
        self.add_staff_button.grid(row=2, column=0, padx=5, pady=5)

        self.remove_student_button = ttk.Button(self.staff_frame, text="Remove Student", command=self.remove_student)
        self.remove_student_button.grid(row=2, column=1, padx=5, pady=5)

    def load_attendance_records(self):
        for item in self.records_table.get_children():
            self.records_table.delete(item)
        attendance = self.load_from_file(attendance_file)
        students = self.load_from_file(students_file)
        for record in attendance:
            student_name = next((student['name'] for student in students if student['id'] == record['student_id']), "Unknown")
            self.records_table.insert("", "end", text=str(record['student_id']),
                                      values=(record['student_id'], student_name, record['date'], 
                                              record.get('check_in', ''), record.get('check_out', ''),
                                              record.get('status', 'Present')))

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

    def load_face_recognition_model(self):
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        self.face_model = Model(inputs=base_model.input, outputs=x)

    def encode_faces(self):
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
            image = cv2.resize(image, (224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)
            image = np.expand_dims(image, axis=0)
            face_encoding = self.face_model.predict(image)[0]
            student['face_encoding'] = face_encoding.tolist()
        self.save_to_file(students_file, students)

    def recognize_faces(self):
        self.recognition_stopped = False
        self.stop_button.config(state="normal")
        self.start_button.config(state="disabled")

        video_capture = cv2.VideoCapture(0)
        students = self.load_from_file(students_file)
        attendance = self.load_from_file(attendance_file)

        while not self.recognition_stopped:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Failed to capture frame from camera")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.detect_faces(rgb_frame)

            for (top, right, bottom, left) in faces:
                face_image = rgb_frame[top:bottom, left:right]
                face_image = cv2.resize(face_image, (224, 224))
                face_image = img_to_array(face_image)
                face_image = preprocess_input(face_image)
                face_image = np.expand_dims(face_image, axis=0)
                face_encoding = self.face_model.predict(face_image)[0]

                name = "Unknown"
                min_dist = float('inf')
                for student in students:
                    if 'face_encoding' not in student:
                        continue
                    dist = np.linalg.norm(face_encoding - np.array(student['face_encoding']))
                    if dist < min_dist:
                        min_dist = dist
                        name = student['name']
                        student_id = student['id']

                if min_dist < 0.6:  # Adjust this threshold as needed
                    self.mark_attendance(student_id, name)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            self.video_label.img = img
            self.video_label.configure(image=img)
            self.video_label.update()

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

        video_capture.release()
        cv2.destroyAllWindows()

    def detect_faces(self, image):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return [(y, x + w, y + h, x) for (x, y, w, h) in faces]

    def mark_attendance(self, student_id, name):
        today = datetime.now().strftime("%Y-%m-%d")
        now_time = datetime.now().strftime("%H:%M:%S")
        attendance = self.load_from_file(attendance_file)

        for record in attendance:
            if record['student_id'] == student_id and record['date'] == today:
                if 'check_out' not in record:
                    record['check_out'] = now_time
                    self.save_to_file(attendance_file, attendance)
                return

        attendance_record = {
            "student_id": student_id,
            "date": today,
            "check_in": now_time,
            "status": "Present"
        }
        attendance.append(attendance_record)
        self.save_to_file(attendance_file, attendance)
        print(f"Marked attendance for {name}")

    def stop_recognition(self):
        self.recognition_stopped = True

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

    def add_staff(self):
        staff_id = self.staff_id_entry.get()
        staff_name = self.staff_name_entry.get()
        if staff_id and staff_name:
            staff = self.load_from_file(staff_file)
            staff.append({"id": staff_id, "name": staff_name})
            self.save_to_file(staff_file, staff)
            messagebox.showinfo("Success", f"Staff {staff_name} added successfully with ID {staff_id}")
            self.staff_id_entry.delete(0, tk.END)
            self.staff_name_entry.delete(0, tk.END)
        else:
            messagebox.showerror("Error", "Staff ID and Name cannot be empty")

    def remove_student(self):
        student_id = messagebox.askstring("Remove Student", "Enter Student ID to remove:")
        if student_id:
            students = self.load_from_file(students_file)
            students = [s for s in students if str(s['id']) != student_id]
            self.save_to_file(students_file, students)
            self.load_attendance_records()
            messagebox.showinfo("Success", f"Student with ID {student_id} removed successfully")
        else:
            messagebox.showerror("Error", "Student ID cannot be empty")

if __name__ == "__main__":
    app = SmartAttendanceApp()
    app.mainloop()
