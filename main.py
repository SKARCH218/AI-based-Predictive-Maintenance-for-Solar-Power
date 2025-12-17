import subprocess
import sys
import time


def run_process(command, name):
    """주어진 명령어를 새 프로세스로 실행하고 출력을 스트리밍합니다."""
    print(f"[{name}] 프로세스 시작: {command}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    # 비블로킹으로 출력 읽기
    def read_output(pipe, prefix):
        for line in iter(pipe.readline, ""):
            print(f"[{prefix}] {line.strip()}")

    # 스레드를 사용하여 stdout과 stderr를 동시에 읽기
    import threading
    stdout_thread = threading.Thread(target=read_output, args=(process.stdout, name))
    stderr_thread = threading.Thread(target=read_output, args=(process.stderr, f"{name}-ERR"))
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()

    return process


if __name__ == "__main__":
    # 필요한 라이브러리 설치 확인
    print("필요한 라이브러리 설치를 시도합니다...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("라이브러리 설치 완료.")
    except subprocess.CalledProcessError as e:
        print(f"라이브러리 설치 중 오류 발생: {e}")
        print("수동으로 'pip install -r requirements.txt'를 실행해 주세요.")
        sys.exit(1)

    api_process = None
    app_process = None
    ai_process = None

    try:
        # api.py 실행 (시리얼 → DB 적재)
        api_process = run_process([sys.executable, "api.py"], "API")
        time.sleep(3)  # api.py 초기화 대기

        # app.py 실행 (웹 서버)
        app_process = run_process([sys.executable, "app.py"], "WEB")

        # server.py 실행 (예측 루프)
        ai_process = run_process([sys.executable, "server.py"], "AI")

        print("\n모든 서비스가 시작되었습니다. 웹 브라우저에서 http://localhost:5000 에 접속하세요.")
        print("종료하려면 Ctrl+C를 누르세요.")

        # 프로세스 상태 모니터링
        while True:
            if api_process.poll() is not None:
                print("[API] 프로세스가 종료되었습니다.")
                break
            if app_process.poll() is not None:
                print("[WEB] 프로세스가 종료되었습니다.")
                break
            if ai_process and ai_process.poll() is not None:
                print("[AI] 프로세스가 종료되었습니다.")
                break
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nCtrl+C 감지. 모든 서비스를 종료합니다...")
    finally:
        if api_process and api_process.poll() is None:
            api_process.terminate()
            api_process.wait(timeout=5)
            if api_process.poll() is None:
                api_process.kill()
            print("[API] 프로세스 종료됨.")

        if app_process and app_process.poll() is None:
            app_process.terminate()
            app_process.wait(timeout=5)
            if app_process.poll() is None:
                app_process.kill()
            print("[WEB] 프로세스 종료됨.")

        if ai_process and ai_process.poll() is None:
            ai_process.terminate()
            ai_process.wait(timeout=5)
            if ai_process.poll() is None:
                ai_process.kill()
            print("[AI] 프로세스 종료됨.")

        print("모든 서비스가 종료되었습니다.")
