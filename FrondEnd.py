import pandas as pd
import tkinter as tk
from tkinter import filedialog, simpledialog
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

class WSNApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detection Of DOS attack in WSN")
        self.root.geometry("800x600")
        self.root.configure(bg="light blue")

        # Title Label
        self.title_label = tk.Label(root, text="Detection Of DOS attack in WSN", font=("Arial", 16, "bold"), bg="light blue", fg="navy")
        self.title_label.pack(pady=10)

        # Frame for buttons
        self.button_frame = tk.Frame(self.root, bg="light blue")
        self.button_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Load Dataset Button
        self.load_data_button = tk.Button(self.button_frame, text="Load Dataset", command=self.load_data, bg="green",
                                          fg="white", width=15)
        self.load_data_button.pack(pady=10)

        # Preprocess Data Button
        self.process_data_button = tk.Button(self.button_frame, text="Preprocess Data", command=self.preprocess_data,
                                             bg="green", fg="white", width=15)
        self.process_data_button.pack(pady=5)

        # Visualize Data Button
        self.visualize_data_button = tk.Button(self.button_frame, text="Visualize Data", command=self.display_visualizations,
                                               bg="green", fg="white", width=15)
        self.visualize_data_button.pack(pady=5)

        # Train Models Button
        self.train_button = tk.Button(self.button_frame, text="Train Models", command=self.train_models,
                                      bg="green", fg="white", width=15)
        self.train_button.pack(pady=5)

        # Test Models Button
        self.test_button = tk.Button(self.button_frame, text="Test Models", command=self.test_models,
                                     bg="green", fg="white", width=15)
        self.test_button.pack(pady=5)

        # Output Frame
        self.output_frame = tk.Frame(self.root, bg="white", width=600, height=400, bd=2, relief=tk.SOLID)
        self.output_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Results Text
        self.results_text = tk.Text(self.output_frame, bg="white", font=("Arial", 10))
        self.results_text.pack(pady=10, fill=tk.BOTH, expand=True)

        # Initialize variables
        self.dataset = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.models = {
            "Decision Tree": DecisionTreeClassifier(),
            "Gaussian Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(n_estimators=100),
            "XGBoost": XGBClassifier()
        }
        self.trained_models = {}

    def load_data(self):
        filename = filedialog.askopenfilename(filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if filename:
            self.dataset = pd.read_csv(filename)
            self.dataset.columns = self.dataset.columns.str.strip()  # Trim column names
            self.display_output("Dataset loaded successfully!", bold=True, color="blue", separator=True)
            self.display_output(f"Shape of the dataset: {self.dataset.shape}", bold=True, color="blue")
            self.display_output("Columns in the dataset:", bold=True, color="blue")
            for column in self.dataset.columns:
                desc = self.get_attribute_description(column)
                self.display_output(f"{column}: {desc}")
            self.display_output("Info of the dataset:", bold=True, color="blue")
            if self.dataset.empty:
                self.display_output("No dataset loaded")
            else:
                self.display_dataset_info()

    def display_dataset_info(self):
        info = "Data types and non-null values:\n"
        for column in self.dataset.columns:
            info += f"{column}: {self.dataset[column].dtype}, {self.dataset[column].count()} non-null values\n"
        self.display_output(info)

    def preprocess_data(self):
        if self.dataset is not None:
            self.dataset['Attack type'] = self.dataset['Attack type'].map({'Normal': 0, 'Grayhole': 1,
                                                                           'Blackhole': 2, 'TDMA': 3, 'Flooding': 4})
            X = self.dataset.drop('Attack type', axis=1)
            y = self.dataset['Attack type']
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            self.display_output("Data preprocessed successfully!", bold=True, color="blue", separator=True)
        else:
            self.display_output("No dataset loaded!", bold=True, color="red")

    def display_visualizations(self):
        if self.dataset is not None:


            # Histogram of Attack Types
            attack_counts = self.dataset['Attack type'].value_counts()
            attack_counts.plot.bar()
            plt.title('Attack Type Histogram')
            plt.xlabel('Attack Type')
            plt.ylabel('Count')
            plt.show()

            # Line plot of Energy Usage over Time
            self.dataset.groupby('Time')['Expaned Energy'].mean().plot() 
            plt.title('Average Energy Usage over Time')
            plt.xlabel('Time')
            plt.ylabel('Average Energy Usage')
            plt.show()

            # Total Data Sent/Received Over Time
            df_summ = self.dataset.groupby("Time")[["DATA_S", "DATA_R"]].sum()
            df_summ.plot.bar(stacked=True)
            plt.title("Total Data Sent/Received Over Time")
            plt.ylabel("Data")
            plt.show()

            # Correlation Heatmap
            sns.clustermap(self.dataset.corr())
            plt.show()

        else:
            self.display_output("No dataset loaded!", bold=True, color="red", separator=True)

    def train_models(self):
        if self.x_train is not None and self.y_train is not None:
            self.trained_models = {}
            for name, model in self.models.items():
                model.fit(self.x_train, self.y_train)
                self.trained_models[name] = model
            self.display_output("Training completed.", bold=True, color="blue", separator=True)
            self.display_output(f"Models trained: {', '.join(self.trained_models.keys())}", bold=True, color="blue")
        else:
            self.display_output("Data not preprocessed yet!", bold=True, color="red", separator=True)

    def test_models(self):
        if not self.trained_models:
            self.display_output("Models not trained yet!", bold=True, color="red", separator=True)
            return

        model_name = simpledialog.askstring("Model Selection", "Enter the model name you want to test:")
        if model_name not in self.trained_models:
            self.display_output("Invalid model name!", bold=True, color="red", separator=True)
            return

        model = self.trained_models[model_name]
        y_pred = model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        classification_rep = classification_report(self.y_test, y_pred)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        self.display_output(f"Model: {model_name}", bold=True, color="blue", separator=True)
        self.display_output(f"Accuracy: {accuracy*100:.2f}%", bold=True, color="blue")
        self.display_output("Classification Report:", bold=True, color="blue")
        self.display_output(classification_rep, bold=False, color="blue")
        self.display_output("Confusion Matrix:", bold=True, color="blue")
        self.display_output(str(conf_matrix), bold=False, color="blue")
        self.display_output("Predictions:", bold=True, color="blue")
        prediction_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for pred in y_pred:
            prediction_count[pred] += 1
        for code, count in prediction_count.items():
            attack_type = self.get_attack_type(code)
            color = "green" if code == 0 else ("yellow" if code == 1 else "red")  # Adjust color based on attack type
            self.display_output(f"{attack_type}: {count}", bold=False, color=color)

    def get_attack_type(self, code):
        attack_types = {0: 'Normal', 1: 'Grayhole', 2: 'Blackhole', 3: 'TDMA', 4: 'Flooding'}
        return attack_types.get(code, 'Unknown')

    def get_attribute_description(self, attribute):
        descriptions = {
            'id': 'Identifier',
            'Time': 'Time of observation',
            'Is_CH': 'Is Cluster Head',
            'who CH': 'Who is Cluster Head',
            'Dist_To_CH': 'Distance to Cluster Head',
            'ADV_S': 'Advanced Setup',
            'ADV_R': 'Advanced Response',
            'JOIN_S': 'Joining Setup',
            'JOIN_R': 'Joining Response',
            'SCH_S': 'Schedule Setup',
            'SCH_R': 'Schedule Response',
            'Rank': 'Rank',
            'DATA_S': 'Data Setup',
            'DATA_R': 'Data Response',
            'Data_Sent_To_BS': 'Data Sent to Base Station',
            'dist_CH_To_BS': 'Distance from Cluster Head to Base Station',
            'send_code': 'Sending Code',
            'Expaned Energy': 'Expanded Energy',
            'Attack type': 'Type of Attack'
        }
        return descriptions.get(attribute, 'Unknown')


    def display_output(self, message, bold=False, color="black", separator=False):
        tag = "bold" if bold else "normal"
        self.results_text.insert(tk.END, message + "\n", color)
        self.results_text.tag_config(color, foreground=color)
        self.results_text.tag_config(tag, font=("Arial", 10, "bold"))
        if separator:
            self.results_text.insert(tk.END, "-" * 50 + "\n", color)

    def tag_text_with_color(self, color):
        self.results_text.tag_config(color, foreground=color)


if __name__ == "__main__":
    root = tk.Tk()
    app = WSNApp(root)
    root.mainloop()
