-------------------------------------------------------------------------------
gridworld example
- run
: python3 gridworld.py --size=4 --gamma=1

default로 size와 gamma값을 각각 4, 1로 가지므로 아래와 같이 실행시 위와 동일함
: python3 Q1.py

k=3, k=10의 value 값을 출력함

-------------------------------------------------------------------------------
Q2
- run
: python3 jacks_car_rental.py --max_cars=20 --rental_cost=10 --move_cost=2 --max_move=5 --upper_bound=11 --gamma=0.9 --avg_requests=[3, 4] --avg_returns=[3, 2]

각 argument들은 default로 위와 같은 값을 가지므로 아래와 같이 실행하여도 무방함
: python3 jacks_car_rental.py

Q2의 경우 main의 143번째 라인을 통해 print할 poliy와 value를 지정할 수 있음
JacksCarRental.policy_iteration(target_policy=[4, 9], target_value=[9])
4, 9번 improved 돼었을 때 policy 출력, 9번 improved 되었을 때 value를 출력하고, 해당 값을 첨부한 jpeg 파일과 동일함