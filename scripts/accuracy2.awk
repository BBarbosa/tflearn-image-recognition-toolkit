BEGIN {
    FS=" ";
    ok=0;
    count=0;
    top=1;
    bot=9;
    print "----- Errors -----"
}

/Class:/ {
    class=$2
    prediction=$5;
    if(class == prediction)
        ok++;
    else
        print "Class ", class, " Prediction ", prediction;
    count++;
}

END {
    if (ok == count)
        print "       None"
    print "------------------"
    print "   ok:", ok;
    print "count:", count;
    print "  acc:", ok/count, "% (", count, "lines)";
}
