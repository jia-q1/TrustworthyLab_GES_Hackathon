import paramiko
import os

# Configuration
hostname = '20.85.126.165'
port = 22
username = 'DatacraftHacker'

# EDIT THESE VARIABLES #########################################

ssh_key_path = 'path_to_openSSH_key' # this is the path to the openSSH key (generate it using puTTYgen)
# note: to get openSSH key: 
# open puTTYgen, load PPK file, go to conversions -> export openSSH key
# don't worry about file extension

local_data_file = 'path_to_local_data_file' # this is the dataset you want to upload on your computer
remote_data_file = 'path_to_remote_data_file' # this is where you want the dataset to be uploaded to
remote_python_script = 'path_to_remote_python_script' # this is Jia's script on the VM
remote_output_file = 'path_to_remote_output_file' # this is the the path to Jia's output result file
local_output_file = 'path_to_local_output_file' # this is where the file ends up on your computer

# ##############################################################

def ssh_connect(hostname, port, username, ssh_key_path):
    # Establish SSH connection using a PPK key.
    k = paramiko.RSAKey.from_private_key_file(filename=ssh_key_path)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, port, username, pkey=k)
    return ssh

def upload_file(sftp, local_path, remote_path):
    # Upload a file to the remote server.
    sftp.put(local_path, remote_path)

def download_file(sftp, remote_path, local_path):
    # Download a file from the remote server.
    sftp.get(remote_path, local_path)

def execute_command(ssh, command):
    # Execute a command on the remote server.
    stdin, stdout, stderr = ssh.exec_command(command)
    return stdout.read(), stderr.read()

# Main execution
if __name__ == "__main__":
    # Connect to the VM
    ssh = ssh_connect(hostname, port, username, ssh_key_path)
    sftp = ssh.open_sftp()

    # Upload the data file
    upload_file(sftp, local_data_file, remote_data_file)

    # Decrypt data
    stdout, stderr = execute_command(ssh, f'python3 /home/DatacraftHacker/summer_hackathon/decrypt.py')
    if stderr:
        print(f"Error: {stderr.decode('utf-8')}")
    else:
        print(f"Output: {stdout.decode('utf-8')}")

    # Execute the analysis script
    stdout, stderr = execute_command(ssh, f'python3 {remote_python_script}')
    if stderr:
        print(f"Error: {stderr.decode('utf-8')}")
    else:
        print(f"Output: {stdout.decode('utf-8')}")

    # Download the generated data file
    download_file(sftp, remote_output_file, local_output_file)

    # Close the connections
    sftp.close()
    ssh.close()

    print("Operation completed successfully.")
