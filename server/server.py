from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import subprocess
import json
import glob
import shutil
import pandas as pd
import joblib
from boosting import ManualVotingEnsemble

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Directories
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
FEATURES_FOLDER = os.path.join(UPLOAD_FOLDER, 'Features_files')
PROCESSED_DATASET = 'output/extracted_features.csv'
PREDICTIONS_OUTPUT = 'output/predictions_output.csv'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FEATURES_FOLDER, exist_ok=True)

# Model Paths
MODEL_PATH = os.path.join(BASE_DIR, 'blended_ensemble_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')

DOCKER_IMAGE_NAME = 'alexmyg/andropytool'

# Class mapping
CLASS_MAPPING = {
    0: "Adware",
    1: "Banking",
    2: "SMS malware",
    3: "Riskware",
    4: "Benign"
}

# Feature Mapping (same as before)
FEATURE_MAPPING = {
            "ACCESS_PERSONAL_INFO___": [
        "android.permission.GET_ACCOUNTS", 
        "android.permission.READ_CONTACTS", 
        "android.permission.READ_CALL_LOG"
    ],
    "CREATE_FOLDER_____": [
        "java.io.File.createNewFile",
        "java.io.File.mkdirs"
    ],
    "CREATE_PROCESS`_____": [
        "java.lang.Runtime.exec"
    ],
    "CREATE_THREAD_____": [
        "java.lang.Thread",
        "java.util.concurrent"
    ],
    "DEVICE_ACCESS_____": [
        "android.permission.ACCESS_WIFI_STATE",
        "android.permission.ACCESS_FINE_LOCATION"
    ],
    "FS_ACCESS____": [
        "java.io.File", 
        "java.io.File.exists", 
        "java.io.File.length"
    ],
    "FS_ACCESS()____": [
        "java.io.File.getPath", 
        "java.io.File.getAbsolutePath"
    ],
    "FS_ACCESS(CREATE)____": [
        "java.io.File.createNewFile", 
        "java.io.File.mkdirs"
    ],
    "FS_ACCESS(CREATE__READ)__": [
        "java.io.FileInputStream", 
        "java.io.BufferedReader", 
        "read"
    ],
    "FS_ACCESS(CREATE__WRITE)__": [
        "java.io.FileOutputStream", 
        "java.io.PrintWriter", 
        "write"
    ],
    "FS_ACCESS(READ)____": [
        "read", 
        "java.io.FileInputStream", 
        "java.io.BufferedReader"
    ],
    "FS_ACCESS(WRITE)____": [
        "write", 
        "java.io.FileOutputStream", 
        "java.io.PrintWriter"
    ],
    "FS_PIPE_ACCESS___": [
        "pipe", 
        "java.io.PipedInputStream", 
        "java.io.PipedOutputStream"
    ],
    "FS_PIPE_ACCESS(READ__WRITE)_": [
        "read", 
        "write", 
        "PipedInputStream", 
        "PipedOutputStream"
    ],
    "FS_PIPE_ACCESS(WRITE)___": [
        "write", 
        "java.io.PipedOutputStream"
    ],
    "NETWORK_ACCESS____": [
        "connect", 
        "socket", 
        "java.net.Socket", 
        "java.net.HttpURLConnection"
    ],
    "NETWORK_ACCESS()____": [
        "socket", 
        "openConnection", 
        "java.net.Socket.open"
    ],
    "NETWORK_ACCESS(READ__WRITE)__": [
        "read", 
        "write", 
        "recv", 
        "send", 
        "socket"
    ],
    "NETWORK_ACCESS(WRITE)____": [
        "send", 
        "write", 
        "java.net.Socket.getOutputStream"
    ],
    "TERMINATE_THREAD": [
        "exit", 
        "kill", 
        "java.lang.Thread.stop", 
        "android.os.Process.killProcess"
    ],
        "__arm_nr_cacheflush": ["__arm_nr_cacheflush"],
    "__arm_nr_set_tls": ["__arm_nr_set_tls"],
    "_llseek": ["llseek", "_llseek"],
    "_newselect": ["select", "_newselect"],
    "access": ["access", "java.io.File.canRead", "java.io.File.canWrite"],
    "addAccessibilityInteractionConnection": [
        "addAccessibilityInteractionConnection", 
        "android.view.accessibility"
    ],
    "addToDisplay": [
        "addToDisplay", 
        "WindowManager.addView"
    ],
    "bind": [
        "bind", 
        "java.net.Socket.bind", 
        "android.app.Service.bindService"
    ],
    "brk": [
        "brk", 
        "java.lang.Runtime.totalMemory"
    ],
    "checkOperation": ["checkOperation"],
    "chmod": [
        "chmod", "java.io.File.setExecutable", "java.io.File.setReadable",
        "Static_analysis.Strings.chmod", "Static_analysis.Strings.chmod%s755 %s",
        "Static_analysis.Strings.chmod 770", "Static_analysis.Strings.chmod 755 %s",
        "Static_analysis.System commands.chmod", "file permissions", "set file access",
        "change file mode", "modify permissions"
    ],
    "clock_gettime": [
        "clock_gettime", "gettimeofday", "time.now", "System.currentTimeMillis", 
        "System.nanoTime", "clock_getres", "fetch time", "system time", 
        "time measurement", "get system clock", "retrieve time", "time syscall", 
        "time tracking", "clock_gettime system call"
    ],
    "clone": [
        "clone", "process clone", "thread.clone", "java.lang.Thread.clone", 
        "java.lang.Object.clone", "java.util.Properties.clone", 
        "java.util.Calendar.clone", "fork", "duplicate thread", "copy process", 
        "spawn thread", "process duplication", "Thread.start", "fork system call", 
        "Thread.newThread", "process fork", "thread creation"
    ],
    "close": [
        "close", 
        "java.io.FileInputStream.close", 
        "java.io.FileOutputStream.close"
    ],
    "connect": [
        "connect", 
        "java.net.Socket.connect", 
        "HttpURLConnection.connect"
    ],
    "dup": ["dup"],
    "epoll_create": ["epoll_create"],
    "epoll_ctl": ["epoll_ctl"],
    "epoll_wait": ["epoll_wait"],
    "execve": [
        "execve", 
        "java.lang.Runtime.exec"
    ],
    "exit": [
        "exit", 
        "System.exit"
    ],
    "exit_group": [
        "exit_group"
    ],
    "fchmod": [
        "fchmod"
    ],
    "fchown32": [
        "fchown32"
    ],
    "fcntl64": [
        "fcntl64"
    ],
    "fdatasync": [
        "fdatasync"
    ],
    "finishDrawing": [
        "finishDrawing"
    ],
    "flock": [
        "flock"
    ],
    "fstat64": [
        "fstat64"
    ],
    "fsync": [
        "fsync"
    ],
    "ftruncate64": ["ftruncate64", "truncate"],
    "futex": ["futex", "synchronization", "java.util.concurrent"],
    "getActiveNetworkInfo": ["getActiveNetworkInfo", "ConnectivityManager", "NetworkInfo"],
    "getActivePhoneType": ["getActivePhoneType", "TelephonyManager.getPhoneType"],
    "getActivityInfo": ["getActivityInfo", "android.content.pm.ActivityInfo"],
    "getApplicationInfo": ["getApplicationInfo", "android.content.pm.ApplicationInfo"],
    "getCellLocation": ["getCellLocation", "TelephonyManager.getCellLocation"],
    "getConnectionInfo": ["getConnectionInfo", "WifiManager.getConnectionInfo"],
    "getDeviceId": ["getDeviceId", "TelephonyManager.getDeviceId"],
    "getDisplayInfo": ["getDisplayInfo", "DisplayManager", "android.view.Display"],
    "getIccSerialNumber": [
        "getIccSerialNumber", 
        "TelephonyManager.getSimSerialNumber", 
        "SIM_SERIAL_NUMBER"
    ],
    "getInTouchMode": [
        "getInTouchMode", 
        "isInTouchMode", 
        "android.view.View.isInTouchMode"
    ],
    "getInputDevice": [
        "getInputDevice", 
        "InputDevice.getDevice", 
        "android.hardware.input.InputDevice"
    ],
    "getInstalledPackages": [
        "getInstalledPackages", 
        "PackageManager.getInstalledPackages", 
        "getPackages", 
        "InstalledPackages"
    ],
    "getInstallerPackageName": [
        "getInstallerPackageName", 
        "PackageManager.getInstallerPackageName", 
        "INSTALLER_PACKAGE"
    ],
    "getLine1Number": [
        "getLine1Number", 
        "TelephonyManager.getLine1Number", 
        "LINE1_NUMBER"
    ],
    "getPackageInfo": [
        "getPackageInfo", 
        "PackageManager.getPackageInfo", 
        "PACKAGE_INFO"
    ],
    "getReceiverInfo": [
        "getReceiverInfo", 
        "PackageManager.getReceiverInfo", 
        "RECEIVER_INFO"
    ],
    "getServiceInfo": [
        "getServiceInfo", 
        "PackageManager.getServiceInfo", 
        "SERVICE_INFO"
    ],
    "getSubscriberId": [
        "getSubscriberId", 
        "TelephonyManager.getSubscriberId", 
        "SUBSCRIBER_ID"
    ],
    "getdents64": ["getdents64", "readdir", "file list"],
    "getpid": ["getpid", "Process.myPid", "android.os.Process.getPid"],
    "getpriority": ["getpriority", "Thread.getPriority"],
    "getsockname": ["getsockname", "Socket.getLocalSocketAddress"],
    "getsockopt": ["getsockopt", "Socket.getOption"],
    "gettid": ["gettid", "Thread.currentThread.getId"],
    "gettimeofday": ["gettimeofday", "System.currentTimeMillis", "System.nanoTime"],
    "getuid32": ["getuid32", "Process.myUid", "android.os.Process.getUid"],
    "hasNavigationBar": ["hasNavigationBar", "ViewConfiguration.hasPermanentMenuKey"],
    "ioctl": ["ioctl"],
    "isAdminActive": ["isAdminActive", "DevicePolicyManager.isAdminActive"],
    "getdents64": [
        "getdents64", "readdir", "file list", "directory read", "file access"
    ],
    "getpid": [
        "getpid", "Process.myPid", "os.getpid", "android.os.Process.getPid", 
        "process id", "PID"
    ],
    "getpriority": [
        "getpriority", "Thread.getPriority", "priority", "process priority", "nice value"
    ],
    "getsockname": [
        "getsockname", "Socket.getLocalSocketAddress", "local socket name", "socket address"
    ],
    "getsockopt": [
        "getsockopt", "Socket.getOption", "socket option", "network option"
    ],
    "gettid": [
        "gettid", "Thread.currentThread.getId", "thread id", "process thread", "TID"
    ],
    "gettimeofday": [
        "gettimeofday", "System.currentTimeMillis", "System.nanoTime", "time", 
        "clock", "date", "get current time"
    ],
    "getuid32": [
        "getuid32", "Process.myUid", "os.getuid", "android.os.Process.getUid", 
        "user id", "UID"
    ],
    "hasNavigationBar": [
        "hasNavigationBar", "ViewConfiguration.hasPermanentMenuKey", "navigation bar", 
        "menu key", "UI navigation"
    ],
    "ioctl": [
        "ioctl", "device control", "input output control", "driver control", "hardware command"
    ],
    "isAdminActive": [
        "isAdminActive", "DevicePolicyManager.isAdminActive", "admin status", 
        "active admin", "device admin", "policy admin"
    ],
        # Strict mappings for identified features
    "isScreenOn": ["android.view.View.isShown"],
    "lseek": ["seek", "llseek", "file.position", "java.io.RandomAccessFile.seek"],
    "lstat64": ["fstat64", "stat"],
    "madvise": ["memory advice", "java.lang.Runtime.gc"],
    "mkdir": ["java.io.File.mkdirs"],
    "mmap2": ["mmap2", "map", "java.nio.MappedByteBuffer"],
    "mprotect": ["mprotect"],
    "munmap": ["unmap", "memory.release", "java.nio.MappedByteBuffer"],
    "nanosleep": ["nanosleep", "java.lang.Thread.sleep"],
    "open": ["open", "java.net.URL.openConnection", "android.content.ContentResolver"],
    "pipe": ["pipe", "java.io.PipedInputStream", "java.io.PipedOutputStream"],
    "poll": ["poll", "select", "java.nio.channels.Selector"],
    "prctl": ["prctl", "process control", "android.os.Process"],
    "pread64": ["pread64", "java.io.RandomAccessFile.read"],
    "pwrite64": ["pwrite64", "java.io.RandomAccessFile.write"],
    # Additional features with close mappings
    "ACCESS_PERSONAL_INFO___": [
        "android.permission.GET_ACCOUNTS", 
        "android.permission.READ_CONTACTS", 
        "android.permission.READ_CALL_LOG"
    ],
     "read": [
        "read", "file.read", "stream.read", "buffered read", "input.read",
        "socket.read", "RandomAccessFile.read", "InputStream.read",
        "BufferedInputStream.read", "FileInputStream.read",
        "readData", "readFile", "data.read", "load", "fetch", "receive",
        "ReadableByteChannel.read", "java.nio.channels.ReadableByteChannel",
        "java.io.DataInput.read", "java.io.Reader.read", "FileChannel.read",
        "input", "fetchData", "recv", "retrieve", "stream", "download",
        "BufferedReader.read", "BufferedInput.read", "socket.receive",
        "getInputStream", "processInput", "fileLoad", "dataLoad", "inputChannel"
    ],
    "recvfrom": ["recvfrom", "socket.receive", "java.net.DatagramPacket.receive"],
    "recvmsg": ["recvmsg", "socket.read", "java.net.Socket.getInputStream"],
    "registerContentObserver": [
        "registerContentObserver", "ContentObserver", "android.database.ContentObserver"
    ],
    "registerInputDevicesChangedListener": [
        "registerInputDevicesChangedListener", "InputManager", "InputDevice"
    ],
    "relayout": ["relayout", "android.view.View.layout", "WindowManager.relayout"],
    "rename": ["rename", "java.io.File.renameTo", "file.rename"],
    "sched_yield": ["sched_yield", "Thread.yield", "process yield", "java.lang.Thread.yield"],
    "send": ["send", "socket.send", "java.net.Socket.getOutputStream"],
    "sendAccessibilityEvent": [
        "sendAccessibilityEvent", "AccessibilityEvent", "android.view.View.sendAccessibilityEvent"
    ],
    "sendmsg": ["sendmsg", "socket.write", "message.send", "java.nio.channels.DatagramChannel.send"],
    "sendto": ["sendto", "socket.sendto", "java.net.DatagramSocket.send"],
    "set": ["set", "java.util.Set", "setProperty", "System.setProperty"],
    "setComponentEnabledSetting": [
        "setComponentEnabledSetting", "android.content.pm.PackageManager.setComponentEnabledSetting"
    ],
    "setInTouchMode": [
        "setInTouchMode", "android.view.View.setInTouchMode", "isInTouchMode"
    ],
    "setpriority": [
        "setpriority", "Thread.setPriority", "process priority", "adjust priority", "sched_yield", "sched_setparam"
    ],
    "setsockopt": [
        "setsockopt", "set socket option", "socket option", "configure socket", 
        "SO_REUSEADDR", "SO_TIMEOUT", "SO_SNDBUF", "SO_RCVBUF"
    ],
    "shutdown": [
        "shutdown", "Socket.shutdown", "shutdownInput", "shutdownOutput", "close socket"
    ],
    "sigaction": [
        "sigaction", "signal action", "handle signal", "signal handler", "configure sigaction"
    ],
    "sigaltstack": [
        "sigaltstack", "signal stack", "alternate signal stack", "setup signal stack", "sigstack", "configure signal"
    ],
    "sigprocmask": [
        "sigprocmask", "signal mask", "block signals", "unblock signals", "mask signal", 
        "sigmask", "signal blocking", "control signals"
    ],
    "socket": [
        "socket", "java.net.Socket", "socket.create", "socket.open", "SocketChannel.open"
    ],
    "stat64": [
        "stat64", "file.stat", "fstat", "file metadata", "file attributes", 
        "file system info", "get file info", "stat syscall"
    ],
    "statfs64": [
        "statfs64", "filesystem stat", "get filesystem info", "disk usage", "file system metadata"
    ],
    "ugetrlimit": [
        "ugetrlimit", "getrlimit", "resource limit", "system resource usage"
    ],
    "unlink": [
        "unlink", "file.unlink", "java.io.File.delete", "delete file", "remove file", "file.remove"
    ],
    "wait4": [
        "wait4", "process.wait", "waitpid", "wait for process", "Thread.join", "process synchronization"
    ],
    "windowGainedFocus": [
        "windowGainedFocus", "onWindowFocusChanged", "WindowManager", "focus event", "window focus"
    ],
    "write": [
        "write", "java.io.OutputStream.write", "java.io.FileOutputStream.write", "BufferedWriter.write",
        "RandomAccessFile.write", "socket.write", "data.write", "stream.write", "file.write"
    ],
    "writev": [
        "writev", "scatter-gather write", "vectorized write", "multi-buffer write", 
        "stream.writev", "buffered output", "data scatter-gather", "file.writev"
    ]
}

def clean_directories():
    """Cleans upload and output directories."""
    try:
        shutil.rmtree(UPLOAD_FOLDER)
        shutil.rmtree(OUTPUT_FOLDER)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(FEATURES_FOLDER, exist_ok=True)
        print("Cleaned directories.")
    except Exception as e:
        print(f"Error cleaning directories: {str(e)}")

def run_andropytool(apk_folder, output_folder):
    """Runs AndroPyTool and processes the APKs."""
    try:
        apk_dir = os.path.abspath(apk_folder).replace("\\", "/")
        output_dir = os.path.abspath(output_folder).replace("\\", "/")

        subprocess.run([
            "docker", "run", "--rm",
            "-v", f"{apk_dir}:/apks",
            "-v", f"{output_dir}:/output",
            DOCKER_IMAGE_NAME, "-s", "/apks/", "--all"
        ], check=True)

        json_files = glob.glob(os.path.join(FEATURES_FOLDER, "*-analysis.json"))
        return json_files
    except Exception as e:
        return {"error": str(e)}

def extract_features(json_files):
    """Extracts features from JSON files."""
    try:
        all_features = []

        for json_file in json_files:
            with open(json_file, 'r') as file:
                json_data = json.load(file)

            features = {feature: 0 for feature in FEATURE_MAPPING}
            json_static = json_data.get('Static_analysis', {})

            for feature, keys in FEATURE_MAPPING.items():
                for key in keys:
                    if key in json_static.get('Permissions', []):
                        features[feature] += 1
                    if key in json_static.get('API calls', {}):
                        features[feature] += json_static['API calls'][key]

            all_features.append(features)

        df = pd.DataFrame(all_features)
        df.to_csv(PROCESSED_DATASET, index=False)
        return PROCESSED_DATASET
    except Exception as e:
        return None

def predict_classes(model_path, scaler_path, encoder_path, input_csv):
    """Predict class labels."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    data = pd.read_csv(input_csv)
    if "Class" in data.columns:
        data = data.drop(columns=["Class"])

    # Scaling input data
    data = pd.DataFrame(scaler.transform(data), columns=data.columns)

    predictions = model.predict(data)
    class_predictions = [CLASS_MAPPING[pred] for pred in predictions]

    output_df = pd.DataFrame({"Predicted_Class": class_predictions})
    output_df.to_csv(PREDICTIONS_OUTPUT, index=False)
    return class_predictions

@app.route('/upload-apk', methods=['POST'])
def upload_apk():
    """Handles APK upload, feature extraction, and class prediction."""
    if 'apk' not in request.files:
        return jsonify({"message": "No file part"}), 400

    file = request.files['apk']
    if file.filename == '':
        return jsonify({"message": "No file selected"}), 400

    if not file.filename.endswith('.apk'):
        return jsonify({"message": "Invalid file type. Please upload an APK file."}), 400

    clean_directories()

    # Save APK
    apk_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(apk_path)

    # Run AndroPyTool
    json_files = run_andropytool(UPLOAD_FOLDER, OUTPUT_FOLDER)
    if not json_files or "error" in json_files:
        return jsonify({"message": "AndroPyTool failed.", "error": str(json_files)}), 500

    # Extract features
    dataset_path = extract_features(json_files)
    if not dataset_path:
        return jsonify({"message": "Failed to extract features."}), 500

    # Predict classes
    predictions = predict_classes(MODEL_PATH, SCALER_PATH, ENCODER_PATH, dataset_path)
    return jsonify({"message": "Prediction completed successfully.", "predictions": predictions})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
