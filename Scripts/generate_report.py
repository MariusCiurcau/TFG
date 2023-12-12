from datetime import datetime
import os

def generate_report(script_code, report, conf_mat, reports_folder):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    txt_filename = f'report_{timestamp}.txt'

    with open(os.path.join(reports_folder, txt_filename), 'w') as txt_file:
        # Write timestamp to file
        txt_file.write(f"Report generated on {timestamp}\n\n")

        # Write script code to file
        txt_file.write("Script Code:\n")
        txt_file.write("-" * 50 + "\n")
        txt_file.write(script_code)
        txt_file.write("\n\n")

        # Write script output to file
        txt_file.write("Classification report:\n")
        txt_file.write("-" * 50 + "\n")
        txt_file.write(report)
        txt_file.write("\n")

        txt_file.write("Confusion matrix:\n")
        txt_file.write("-" * 50 + "\n")
        txt_file.write(conf_mat)

    print(f"Text report generated: {txt_filename}")

