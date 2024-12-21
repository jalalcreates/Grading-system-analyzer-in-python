import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random


columns = ['StudentID', 'Name', 'Maths', 'Science', 'English', 'History', 'Grade', 'Attendance', 'Pass/Fail', 'Eligible for Exam']
data = [
    [1, 'Ubaid', 85, 92, 78, 88, 'A', 95, 'Pass', 'Eligible'],
    [2, 'Jalal', 75, 80, 68, 65, 'B', 80, 'Pass', 'Eligible'],
    [3, 'Jawad', 95, 98, 92, 96, 'A', 100, 'Pass', 'Eligible'],
    [4, 'Taimur', 60, 65, 55, 58, 'C', 85, 'Fail', 'Eligible'],
    [5, 'Mubeen', np.nan, 80, 85, 78, 'B', 90, 'Pass', 'Eligible'],
    [6, 'Zaid', 88, 87, 85, np.nan, 'B', 70, 'Pass', 'Not Eligible'],
    [7, 'Shujaat', 70, 72, 78, 69, 'B', 95, 'Pass', 'Eligible'],
]


df = pd.DataFrame(data, columns=columns)


def add_student(df, student_id, name, maths, science, english, history, grade, attendance):
    new_student = pd.DataFrame([[student_id, name, maths, science, english, history, grade, attendance, '', '']], columns=df.columns)
    df = pd.concat([df, new_student], ignore_index=True)
    print(f"Added student {name} with ID {student_id}.")
    return df


def update_student_grade(df, student_id, subject, new_grade):
    if subject not in ['Maths', 'Science', 'English', 'History']:
        print(f"Invalid subject: {subject}. Please enter a valid subject.")
        return df
    
    student_index = df[df['StudentID'] == student_id].index
    if len(student_index) == 0:
        print(f"No student found with ID {student_id}.")
        return df
    
    df.loc[student_index, subject] = new_grade
    print(f"Updated {subject} grade for student {student_id} to {new_grade}.")
    return df


def save_data_to_csv(df, filename="students_data.csv"):
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


def load_data_from_csv(filename="students_data.csv"):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        print("No saved data found, using sample data.")
        return df


def relative_grading(df):
    for subject in ['Maths', 'Science', 'English', 'History']:
        subject_avg = df[subject].mean()
        df['Grade'] = df.apply(lambda row: 'A' if row[subject] >= subject_avg + 10 else 'B' if row[subject] >= subject_avg else 'C', axis=1)
    print("Relative grading applied based on subject averages.")
    return df


def apply_pass_fail(df, pass_threshold=50):
    for subject in ['Maths', 'Science', 'English', 'History']:
        df[subject + ' Pass/Fail'] = df[subject].apply(lambda x: 'Pass' if x >= pass_threshold else 'Fail')
    print("Pass/Fail status updated based on grade threshold.")
    return df


def apply_exam_eligibility(df, attendance_threshold=80):
    df['Eligible for Exam'] = df['Attendance'].apply(lambda x: 'Eligible' if x >= attendance_threshold else 'Not Eligible')
    print("Exam eligibility based on attendance applied.")
    return df


def grade_distribution(df):
    grade_count = df['Grade'].value_counts()
    print(f"\nGrade Distribution:\n{grade_count}")
    grade_count.plot(kind='bar', title="Grade Distribution", color='lightblue')
    plt.xlabel("Grade")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.show()


def subject_performance_distribution(df):
    subjects = ['Maths', 'Science', 'English', 'History']
    plt.figure(figsize=(12, 8))
    for i, subject in enumerate(subjects, 1):
        plt.subplot(2, 2, i)
        df[subject].dropna().hist(bins=10, color='skyblue', edgecolor='black')
        plt.title(f'{subject} Score Distribution')
        plt.xlabel('Score')
        plt.ylabel('Number of Students')
    plt.tight_layout()
    plt.show()


def pass_fail_distribution(df):
    subjects = ['Maths', 'Science', 'English', 'History']
    for subject in subjects:
        pass_fail = df[subject + ' Pass/Fail'].value_counts()
        pass_fail.plot(kind='bar', title=f"Pass/Fail Distribution in {subject}", color=['green', 'red'])
        plt.xlabel('Pass/Fail')
        plt.ylabel('Number of Students')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()


def attendance_vs_scores(df, subject):
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Attendance'], df[subject], color='blue', alpha=0.7)
    plt.title(f'Attendance vs {subject} Scores')
    plt.xlabel('Attendance (%)')
    plt.ylabel(f'{subject} Score')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def overall_performance_heatmap(df):
    performance_data = df[['Maths', 'Science', 'English', 'History']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(performance_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Student Performance')
    plt.tight_layout()
    plt.show()


def top_n_students_per_subject(df, n=3):
    subjects = ['Maths', 'Science', 'English', 'History']
    for subject in subjects:
        top_students = df.nlargest(n, subject)[['Name', subject]]
        top_students.plot(kind='bar', x='Name', y=subject, title=f"Top {n} Students in {subject}", color='lightgreen')
        plt.xlabel('Student')
        plt.ylabel(f'{subject} Score')
        plt.tight_layout()
        plt.show()


def class_ranking(df):
    df['Overall'] = df[['Maths', 'Science', 'English', 'History']].mean(axis=1)
    df['Rank'] = df['Overall'].rank(ascending=False)
    print("\nClass Ranking:")
    print(df[['Name', 'Overall', 'Rank']].sort_values(by='Rank'))
    return df


def subject_statistics(df):
    subjects = ['Maths', 'Science', 'English', 'History']
    stats = pd.DataFrame(columns=['Subject', 'Average', 'Median', 'Std Dev'])
    for subject in subjects:
        avg = df[subject].mean()
        median = df[subject].median()
        std_dev = df[subject].std()
        stats = stats._append({'Subject': subject, 'Average': avg, 'Median': median, 'Std Dev': std_dev}, ignore_index=True)
    print("\nSubject-wise Statistics:")
    print(stats)
    return stats


def attendance_improvement(df):
    df['Attendance Improvement'] = np.random.uniform(0, 10, size=len(df))  
    print("\nAttendance Improvement Data:")
    print(df[['Name', 'Attendance Improvement']])


def compare_students(df, student_id1, student_id2):
    student1 = df[df['StudentID'] == student_id1].iloc[0]
    student2 = df[df['StudentID'] == student_id2].iloc[0]
    comparison = pd.DataFrame({
        'Subject': ['Maths', 'Science', 'English', 'History'],
        student1['Name']: [student1['Maths'], student1['Science'], student1['English'], student1['History']],
        student2['Name']: [student2['Maths'], student2['Science'], student2['English'], student2['History']]
    })
    print(f"\nPerformance Comparison between {student1['Name']} and {student2['Name']}:")
    print(comparison)


def top_performers_in_multiple_subjects(df, subjects=['Maths', 'Science', 'English']):
    df['Top Performer'] = df[subjects].apply(lambda row: row.idxmax(), axis=1)
    top_performers = df.groupby('Top Performer')['Name'].apply(list)
    print("\nTop Performers in Multiple Subjects:")
    print(top_performers)




def track_attendance(df, student_id, attendance_percentage):
    student_index = df[df['StudentID'] == student_id].index
    if len(student_index) == 0:
        print(f"No student found with ID {student_id}.")
        return df
    
    df.loc[student_index, 'Attendance'] = attendance_percentage
    print(f"Updated attendance for student {student_id} to {attendance_percentage}%.")
    return df


def main():
    global df
    while True:
        print("\n--- School Management System ---")
        print("1. Add Student")
        print("2. Update Student Grade")
        print("3. Track Student Attendance")
        print("4. Apply Relative Grading")
        print("5. Apply Pass/Fail Based on Grades")
        print("6. Apply Exam Eligibility Based on Attendance")
        print("7. Generate Grade Distribution")
        print("8. Visualize Subject-wise Performance Distribution")
        print("9. Visualize Pass/Fail Distribution per Subject")
        print("10. Visualize Attendance vs Scores")
        print("11. Visualize Overall Performance Heatmap")
        print("12. Visualize Top N Students per Subject")
        print("13. Class Ranking")
        print("14. Show Subject-wise Statistics")
        print("15. Track Attendance Improvement")
        print("16. Compare Two Students' Performances")
        print("17. Identify Top Performers in Multiple Subjects")
        print("18. Save Data to CSV")
        print("19. Load Data from CSV")
        print("20. Exit")

        choice = int(input("Enter your choice: "))

        if choice == 1:
            student_id = int(input("Enter Student ID: "))
            name = input("Enter Student Name: ")
            maths = float(input("Enter Maths grade: "))
            science = float(input("Enter Science grade: "))
            english = float(input("Enter English grade: "))
            history = float(input("Enter History grade: "))
            grade = input("Enter Grade: ")
            attendance = float(input("Enter Attendance percentage: "))
            df = add_student(df, student_id, name, maths, science, english, history, grade, attendance)
        
        elif choice == 2:
            student_id = int(input("Enter Student ID: "))
            subject = input("Enter Subject to update: ")
            new_grade = float(input("Enter new grade: "))
            df = update_student_grade(df, student_id, subject, new_grade)
        
        elif choice == 3:
            student_id = int(input("Enter Student ID: "))
            attendance_percentage = float(input("Enter new attendance percentage: "))
            df = track_attendance(df, student_id, attendance_percentage)
        
        elif choice == 4:
            df = relative_grading(df)
        
        elif choice == 5:
            df = apply_pass_fail(df)
        
        elif choice == 6:
            df = apply_exam_eligibility(df)
        
        elif choice == 7:
            grade_distribution(df)
        
        elif choice == 8:
            subject_performance_distribution(df)
        
        elif choice == 9:
            pass_fail_distribution(df)
        
        elif choice == 10:
            subject = input("Enter Subject for attendance vs scores plot: ")
            attendance_vs_scores(df, subject)
        
        elif choice == 11:
            overall_performance_heatmap(df)
        
        elif choice == 12:
            n = int(input("Enter the number of top students to visualize: "))
            top_n_students_per_subject(df, n)
        
        elif choice == 13:
            df = class_ranking(df)
        
        elif choice == 14:
            subject_statistics(df)
        
        elif choice == 15:
            attendance_improvement(df)
        
        elif choice == 16:
            student_id1 = int(input("Enter first Student ID: "))
            student_id2 = int(input("Enter second Student ID: "))
            compare_students(df, student_id1, student_id2)
        
        elif choice == 17:
            top_performers_in_multiple_subjects(df)
        
        elif choice == 18:
            save_data_to_csv(df)
        
        elif choice == 19:
            filename = input("Enter filename to load data from: ")
            df = load_data_from_csv(filename)
        
        elif choice == 20:
            break


if __name__ == "__main__":
    main()
