import os
import datetime
import subprocess
import pandas as pd
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import torch
import Utils
import torchvision.transforms as transforms
from torchvision.models import resnet18

class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")

        image_path = "image.jpg"  # Update with the actual path to your image
        self.image = Image.open(image_path)
        self.image = self.image.resize((400, 200))
        self.image = ImageTk.PhotoImage(self.image)
        self.image_label = tk.Label(self.main_window, image=self.image)
        self.image_label.place(x=30, y=50)

        self.login_button_main_window = Utils.get_button(self.main_window, 'Mark Attendance', 'sky blue', self.login, fg='black')
        self.login_button_main_window.place(x=30, y=300)

        self.register_new_user_button_main_window = Utils.get_button(self.main_window, 'Register New Student', 'gray',
                                                                          self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=30, y=400)

        self.webcam_label = Utils.get_img_label(self.main_window)
        self.webcam_label.place(x=450, y=0, width=700, height = 500)

        self.add_webcam(self.webcam_label)

        self.db_dir = './Student Images'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.attendance_file = 'Attendance.xlsx'
        self.student_info_file = 'Student Information.xlsx'

        # Load the pre-trained ResNet model
        self.model = resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 10)  # Adjust final fully connected layer
        self.model.load_state_dict(torch.load('resnet_model.pth', map_location=torch.device('cpu')))
        self.model.eval()  # Set the model to evaluation mode

        # Define the transformation for input images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)

        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        self.most_recent_capture_arr = frame

        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)

        imgtk = ImageTk.PhotoImage(image= self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)

        self._label.after(20, self.process_webcam)

    def login(self):
        unknown_img_path = './.tmp.jpg'
        cv2.imwrite(unknown_img_path, self.most_recent_capture_arr)
        name = self.recognize_face(unknown_img_path)

        if name == 'unknown_person':
            Utils.msg_box('Oops...', 'Unknown User \nPlease Register as a New User or try again.')
        else:
            self.record_attendance(name)
            Utils.msg_box('Welcome!', 'Welcome, {}'.format(name))

        os.remove(unknown_img_path)

    def recognize_face(self, image_path):
        # Preprocess the clicked image
        img = Image.open(image_path)
        img = self.transform(img)
        img = img.unsqueeze(0)  # Add batch dimension

        # Pass the image through the model to get predictions
        with torch.no_grad():
            outputs = self.model(img)
            _, predicted = torch.max(outputs, 1)
            class_index = predicted.item()

        # Map the predicted class index to class name
        class_name = self.get_class_name(class_index)
        return class_name

    def get_class_name(self, class_index):
        # Define your mapping from class index to class name
        class_names = ['Abir', 'Fahim', 'Hemel', 'Nipa', 'Rupak', 'Sadiqul', 'Shepon', 'Tama', 'Tamim', 'Tarup']

        # Return the class name corresponding to the class index
        if 0 <= class_index < len(class_names):
            return class_names[class_index]
        else:
            return 'unknown_person'

    def record_attendance(self, name):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        attendance_data = pd.DataFrame({'Date Time': [now], 'Name': [name]})
        try:
            student_info = pd.read_excel(self.student_info_file)
            student_info = student_info.loc[student_info['Name'] == name]
            if not student_info.empty:
                student_id = student_info.iloc[0]['Student ID']
                attendance_data['Student ID'] = student_id
                if os.path.exists(self.attendance_file):
                    existing_data = pd.read_excel(self.attendance_file)
                    attendance_data = pd.concat([existing_data, attendance_data], ignore_index=True)
                attendance_data.to_excel(self.attendance_file, index=False)
            else:
                raise ValueError("Student information not found for {}".format(name))
        except Exception as e:
            print("Error saving attendance:", e)

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")

        self.accept_button_register_new_user_window = Utils.get_button(self.register_new_user_window, 'Accept', 'sky blue', self.accept_register_new_user, fg='black')
        self.accept_button_register_new_user_window.place(x=30, y=300)

        self.try_again_button_register_new_user_window = Utils.get_button(self.register_new_user_window, 'Try Again', 'red',self.accept_register_new_user)
        self.try_again_button_register_new_user_window.place(x=30, y=400)

        self.capture_label = Utils.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=450, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = Utils.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=30, y=150)

        self.text_label_register_new_user = Utils.get_text_label(self.register_new_user_window, 'Please, \nInput Username:')
        self.text_label_register_new_user.place(x=30, y=70)

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c")

        cv2.imwrite(os.path.join(self.db_dir, '{}.jpg'.format(name)), self.register_new_user_capture)

        Utils.msg_box('Success!','User was Registered Successfully')

        self.register_new_user_window.destroy()

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def start(self):
        self.main_window.mainloop()

if __name__ == "__main__":
    app = App()
    app.start()