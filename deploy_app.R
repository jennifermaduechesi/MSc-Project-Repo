# Deploy the violence risk interface to the student's own shinyapps.io account.
#
# Run at the student's explicit instruction, with credentials they supplied for this
# purpose and said they would rotate afterwards. The bundle is the `app` folder: the
# interface, its stylesheet, and the scored tables it reads.
#
# The student was asked whether to include `data/recent_incidents.csv`, which carries
# incident dates, locations, casualty counts and perpetrator attribution derived from the
# Bulwark Intelligence corpus, and chose to include it.
library(rsconnect)

rsconnect::deployApp(
  appDir         = "/home/user/MSc-Project-Repo/app",
  appName        = "lga-violence-risk-3window",
  appTitle       = "Seven, fourteen and twenty-eight day violence risk, Nigerian LGAs",
  account        = "jenniferchidi",
  server         = "shinyapps.io",
  forceUpdate    = TRUE,
  launch.browser = FALSE,
  logLevel       = "normal"
)
