#!/bin/sh

sleep 600
while true #runs forever
do
	a=`cut -d ' ' -f 3 /proc/loadavg` #set a = load avg.
	if [ $(echo "$a > 0.5" | bc) -ne 0 ]; then #If load avg. is less than .25
		sudo shutdown #Suspend
	else #If it's greater than .25
		sleep 600 #Wait ten minutes (600 seconds) and try again
	fi
done