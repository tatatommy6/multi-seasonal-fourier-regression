## Shape Test for MSFR - Prototype
- Dataset : None
- Reason why we should do this Benchmark : MSFR에 대한 Tensor Shape가 비확실 하다고 판단 
- Tester : 함태준(rrayy-25809)

Tensor Shape 확인 결과, 제대로 Flatten 되고 있는 것을 확인함, 하지만 학습이 원활하지 않고, 파라미터 학습 비율이 생각보다 저조함.
초기화 함수로 초기화 한 값에서 거의 변하지 않아 선형회귀의 초기화 함수를 사용하였으나, 결과는 달라지지 않음

결론적으로 Shape 확인에는 성공했으나, 학습이 원활하지 않는다는 이슈를 발견함.

그리하여 해당 Benchmark를 수정하여, 학습이 원활하게 돌아갈 수 있도록 수정할 수 있도록 테스트 하는 Benchmark로 수정할 예정
