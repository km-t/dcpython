import outBalancedLogsbyAngle as a
import outBalancedLogsbyPower as p
import outBalancedLogsbyWhere as w
import changeLog as c
import namedLogToVector as n
file = "../logs/namedLogs.csv"
# n.main()
a.main(file)
p.main(file)
w.main(file)
c.main(file)
