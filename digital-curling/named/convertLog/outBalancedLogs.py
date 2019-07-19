import outBalancedLogsbyAngle as a
import outBalancedLogsbyPower as p
import outBalancedLogsbyWhere as w
import changeLog as c
import namedLogToVector as n
import addTurn as ad
file = "../logs/namedLogs.csv"
file2 = "../logs/namedLogsWithTurn.csv"
ad.main(file)
n.main()
a.main(file2)
p.main(file2)
w.main(file2)
c.main(file2)
