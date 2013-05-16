# Sup and unsup learning for street environment understanding

## Dataset architecture

### INRIA (person)

Training dataset size: 1 208

Testing dataset size: 563

\# categories: 1

### GTSRB (sign)

Training dataset size: 13 070
![alt text][pin]

Testing dataset size: 12 630

\# categories: 43

![alt text][pin] *Just the last 10 frames of every physical sign have been used, because meaningful.* 
*Total amount of training images: 3 × 13070 = 39 210.*

#### Category dimensionality

The training dataset is made of 43 *signs*, which has multiple (7 to 75) corresponding *physical signs*, recorded for 1s, @30fps (= 30 photograms). 
For evert category (1 to 43) the number of *physical signs* and its description is reported below. 
For example, the number of training images of the **20 km/h speed limit** is: 7 × 30 = 210. 
For the **50 km/h speed limit** we have instead: 75 × 30 = 2 250.

1. 7  - 20 km/h speed limit
2. 74 - 30 km/h speed limit
3. 75 - 50 km/h speed limit
4. 47 - 60 km/h speed limit
5. 66 - 70 km/h speed limit
6. 62 - 80 km/h speed limit
7. 14 - 80 km/h end of speed limit
8. 48 - 100 km/h speed limit
9. 47 - 120 km/h speed limih
10. 49 - No passing
11. 67 - No passing for vehicles over 3.5t
12. 44 - Priority
13. 70 - Priority road
14. 72 - Yield
15. 26 - Stop
16. 21 - Prohibited for all vehicles
17. 14 - Vehicles over 3.5t prohibited
18. 37 - Do not enter
19. 40 - General danger
20. 7 - Curve (left)
21. 12 - Curve (right)
22. 11 - Double curve. First curve is to the left
23. 13 - Rough road
24. 17 - Slippery when wet or dirty
25. 9 - Road narrows (right side)
26. 50 - Road work
27. 20 - Traffic signals ahead
28. 8 - Pedestrians
29. 18 - Watch for children
30. 9 - Bicycle crossing
31. 15 - Beware of ice/snow
32. 26 - Wild animal crossing
33. 8 - End of all restrictions
34. 23 - Mandatory direction of travel. All traffic must turn right
35. 14 - Mandatory direction of travel. All traffic must turn left
36. 40 - Mandatory direction of travel. All traffic must continue straight ahead (i.e. no turns)
37. 13 - Mandatory direction of travel. All traffic must continue straight ahead or turn right (i.e. no left turn)
38. 7 - Mandatory direction of travel. All traffic must continue straight ahead or turn left (i.e. no right turn)
39. 69 - Pass by on right
40. 10 - Pass by on left
41. 12 - Roundabout
42. 8 - End of no passing zone
43. 8 - End of no passing zone for vehicles over 3.5t

### KITTI (car)

Training dataset size: 6 427

Testing dataset size: 1 607

\# categories: 1

[pin]: http://icons.iconarchive.com/icons/icons-land/vista-map-markers/16/Map-Marker-Push-Pin-1-Right-Azure-icon.png
