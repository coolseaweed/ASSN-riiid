from pydantic import BaseModel
from typing import Literal, List


class StudentInfo(BaseModel):
    gender: Literal["Boy", "Girl"]
    age: Literal["21-25", "16-20", "11-15", "26-30", "6-10", "1-5"]
    education_level: Literal["University", "College", "School"]
    institution_type: Literal["Non Government", "Government"]
    it_student: Literal["No", "Yes"]
    location: Literal["Yes", "No"]
    load_shedding: Literal["Low", "High"]
    financial_condition: Literal["Mid", "Poor", "Rich"]
    internet_type: Literal["Wifi", "Mobile Data"]
    network_type: Literal["4G", "3G", "2G"]
    class_duration: Literal["3-6", "1-3", "0"]
    self_lms: Literal["No", "Yes"]
    device: Literal["Tab", "Mobile", "Computer"]


class PredictResponse(BaseModel):
    adaptivity_level: Literal["Moderate", "Low", "High"]
