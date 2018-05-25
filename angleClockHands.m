function a = angleClockHands(hour,mins)

h = mod(hour,12);
m = mod(mins,60);

ha = 0.5 * (h*60 + m);
ma = 6*m;

a1 = abs(ha - ma);

a = min(360-a1, a1);

end

