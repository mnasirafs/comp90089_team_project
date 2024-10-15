package edu.gatech.comp90089.model

import java.sql.Timestamp

case class ICURecord(icustay_id: Int, intime: Timestamp, outtime: Timestamp, suspected_infection_time: Timestamp)
case class SOFA(icustay_id: Int, hr: Int, starttime: Timestamp, endtime: Timestamp, sofa: Int)
