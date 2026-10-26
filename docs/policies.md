# Policy 비교 후보

아래는 도입 후보이며 현재 저장소에 구현되거나 FFW-SG2에서 검증된 policy가 아닙니다. 실제 backend 지원, 장비의 관측/행동 규격, 학습 자원을 확인한 뒤 선택합니다.

| 후보 | 비교 목적 | 현재 상태 |
| --- | --- | --- |
| Teleoperation / scripted baseline | 작업 가능성과 기록·평가 흐름 확인. Teleoperation 결과는 자율 policy 순위와 별도로 보고 | 계획 |
| ACT | 자체 시연 데이터로 첫 학습 baseline 준비 | 미구현 |
| Diffusion | 같은 데이터와 평가 조건에서 다른 행동 생성 방식 비교 | 미구현 |
| SmolVLA | 사전학습 모델 기반 미세조정 및 언어 조건을 사용하는 비교 후보 | 미구현 |

## 도입 순서 제안

작업 가능성 확인 → ACT → Diffusion → SmolVLA 순으로 진행합니다. 한 후보가 기록·학습·평가 전 과정을 통과한 뒤 다음 후보를 추가합니다. 사전학습 여부, 데이터, 입력, 연산 예산이 다르면 차이를 명시합니다.

ROBOTIS [Model Training Guide](https://docs.robotis.com/docs/systems/aiworker/imitation_learning/model_training/)는 ACT, Diffusion, SmolVLA 등을 지원 모델로 나열하고, 첫 데이터셋에서 선택이 불분명하면 ACT로 시작하도록 안내합니다. 이는 공식 도구의 지원 목록이며 이 연구의 성능 순위나 실기 호환성 검증 결과가 아닙니다. 확인일: 2026-09-29.

## 후보별로 확정할 사항

- 구현 저장소·commit·라이선스와 pretrained checkpoint의 출처·revision·checksum
- 실제 관측 key, 카메라 순서·크기, 상태·행동 단위와 joint 순서
- 관측 history, action horizon, 실행 주기, 정규화 통계
- train/validation/test split, 학습 seed와 예산, 모델 선택 기준
- 실행 지연, 행동 한계, 오류·중단 처리, 실험별 평가 결과

학습·추론 코드 연결 시 각 policy의 입출력을 같은 로봇 인터페이스에 맞춥니다. 양식의 값이 채워지기 전에는 설정을 바로 로봇에 전달하지 않습니다.
