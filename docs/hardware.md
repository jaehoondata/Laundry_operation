# 장비 및 실행 환경

사용자가 지정한 플랫폼은 **ROBOTIS AI Worker FFW-SG2**입니다. 아래 값은 실제 장비에서 확인한 뒤 기록합니다. 모델 이름만으로 카메라 설정, joint 순서, 드라이버 버전을 확정하지 않습니다.

| 항목 | 현재 기록 |
| --- | --- |
| 플랫폼 | ROBOTIS AI Worker FFW-SG2 |
| 실행 환경 | TBD: 실제 로봇 / 시뮬레이터 |
| 하드웨어 revision / 그리퍼 구성 | TBD |
| 로봇 PC OS / ROS 배포판 / RMW | TBD |
| `ai_worker` commit / 컨테이너 digest | TBD |
| 학습·추론 backend / commit | TBD |
| 학습 PC / GPU / CUDA / Python | TBD |
| 카메라별 이름 / 해상도 / FPS / timestamp 기준 | TBD |
| 카메라 내부·외부 보정 파일 revision | TBD |
| 관측 key / shape / 단위 / joint 순서 | TBD |
| 행동 표현 / shape / 단위 / joint 순서 / 주기 | TBD |
| 양팔·그리퍼 사용 범위 | TBD |
| base / lift / head 운용 조건 | TBD |
| 작업대 크기·높이 / 작업 영역 | TBD |
| 정지·재시작 절차 / 동작 한계 | TBD |

첫 작업에서는 이동 base와 작업대의 상대 위치를 일정하게 유지하는 실험을 제안합니다. 구체적인 제어 방법과 제한은 장비 설정을 확인한 뒤 정합니다. 장비 주소·비밀번호 등 접속 정보는 로컬에만 보관합니다.

## 공식 자료

2026-09-29 확인. 문서의 현재 권장 구성과 실제 설치 버전이 다를 수 있으므로 실행 환경은 commit과 image digest로 기록합니다.

- [AI Worker 공식 오픈소스 안내](https://docs.robotis.com/docs/systems/aiworker/resources/open_source/): 로봇 구동, 모방학습, 제어, 시뮬레이션 저장소 목록.
- [AI Worker 하드웨어](https://ai.robotis.com/ai_worker/hardware_ai_worker): FFW-SG2 제품 사양과 작업 영역.
- [ROBOTIS-GIT/ai_worker](https://github.com/ROBOTIS-GIT/ai_worker): 로봇 플랫폼 소프트웨어.
- [ROBOTIS-GIT/cyclo_intelligence](https://github.com/ROBOTIS-GIT/cyclo_intelligence): 시연 기록, 데이터 준비, 학습·추론 작업 흐름.
- [ROBOTIS-GIT/cyclo_lab](https://github.com/ROBOTIS-GIT/cyclo_lab): 시뮬레이션 자산 및 환경.

이 저장소에는 장비를 움직이는 launch 명령이나 미검증 joint 명령을 넣지 않았습니다. 연결 단계에서 설치 버전에 맞는 공식 절차와 관측/행동 계약을 먼저 확인합니다.
