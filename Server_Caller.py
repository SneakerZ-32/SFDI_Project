import paramiko
import scp
import io
import sys

def run_on_remote_server(server_ip, username, password, local_script_path, remote_script_path):
    # Initialize the SSH client
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect to the server
        ssh_client.connect(hostname=server_ip, username=username, password=password)
        print('Connection to server established.')

        # Create an SCP client
        with scp.SCPClient(ssh_client.get_transport()) as scp_client:
            # Upload the script to the server
            scp_client.put(local_script_path, remote_script_path)
            print(f"Uploaded {local_script_path} to {remote_script_path}")

        # Run the script on the server
        stdin, stdout, stderr = ssh_client.exec_command(f'python {remote_script_path}')

        # Capture and print the output
        for line in stdout:
            print(line.strip())

        # Check for any errors
        for line in stderr:
            print(f"Error: {line.strip()}")

        # Download the results (if any)
        # with scp.SCPClient(ssh_client.get_transport()) as scp_client:
        #     scp_client.get('remote_results.txt', 'local_results.txt')
        #     print("Downloaded results from the server")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        ssh_client.close()
        print("Connection closed")

# Example usage
if __name__ == "__main__":
    server_ip = '192.168.2.19'
    username = 'nturcan'
    password = input('Password for server: ')
    local_script_path = 'OriginalTutorialPytorch.py'
    remote_script_path = 'home/desktop/Solver Example Code/originalTutorialPytorch.py'

    run_on_remote_server(server_ip, username, password, local_script_path, remote_script_path)
