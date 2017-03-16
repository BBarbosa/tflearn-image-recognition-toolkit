BEGIN {
    FS=" ";
    sum=0;
    count=0;
    top=1;
    bot=9;
    max=0;
    min=100;
}

/Total:/ {
    print "   ",$10;
    sum += $10;
    count++;
}

END {
    print "-------------"
    print "avg", sum/count, "% (", count, "lines)";
}
