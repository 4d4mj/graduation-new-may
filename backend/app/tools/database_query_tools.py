import logging
from typing import Optional, Annotated, List
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from app.db.session import tool_db_session
from app.db.crud.patient import (
    find_patients_by_name_and_verify_doctor_link,
    get_patients_for_doctor,
)
from app.db.models import PatientModel

logger = logging.getLogger(__name__)

@tool("get_patient_info")
async def get_patient_info(patient_full_name: str, user_id: Annotated[int, InjectedState("user_id")]) -> str:
    """
    Fetches basic demographic information (Date of Birth, sex, phone number, address)
    for a specific patient if they have an appointment record with the requesting doctor
    The patient_full_name should be the first and last name of the patient 

    """
    logger.info(f"Tool 'get_patient_info' invoked by doctor_id '{user_id}' for patient patient: '{patient_full_name}'")
    
    if not patient_full_name or not patient_full_name.strip():
        return "Please provide the full name of the patient you are looking for"
    
    async with tool_db_session() as db:
        try:
            #user_id here is the requesting_doctor_id from the agent's state
            patients = await find_patients_by_name_and_verify_doctor_link(
                db, full_name=patient_full_name, requesting_doctor_id=user_id
            )
            
            if not patients:
                return f"'No patient named '{patient_full_name}' found with an appointment record associated with you"
            
            if len(patients) > 1:
                # If multiple patients with the same name are linked to this doctor,
                # provide enough info for the doctor (via LLM) to disambiguate.
                response_lines = [
                    f"Multiple patients named '{patient_full_name}' found who have had appointments with you. Please specify using their date of birth: "
                ]
                for p in patients:
                    dob_str = p.dob.strftime('%Y-%m-%d') if p.dob else 'DOB not available'
                    response_lines.append(f"- {p.first_name} {p.last_name} (DOB: {dob_str})")
                
                return "\n".join(response_lines)
            
            # Exact;y onr patient found
            patient = patients[0]
            dob_str = patient.dob.strftime('%Y-%m-%d') if patient.dob else 'N/A'
            sex_str = patient.sex or "N/A"
            phone_str = patient.phone or "N/A"
            address_str = patient.address or "N/A"
            
            return (f"Patient Information for {patient.first_name} {patient.last_name}:\n"
                    f"- Date of Birth: {dob_str}\n"
                    f"- Sex: {sex_str}\n"
                    f"- Phone: {phone_str}\n"
                    f"- Address: {address_str}")
        
        except Exception as e:
            logger.error(
                f"Tool: 'get_patient_info': Error processing request for doctor_id '{user_id}', patient '{patient_full_name}': {e}",
                exc_info=True
            )                
            return "An unexpected error occurred while trying to retrieve patient information. Please try again later."
                

@tool("list_my_patients")
async def list_my_patients(user_id: Annotated[int, InjectedState("user_id")], page: Optional[int] = 1, page_size: Optional[int] = 10) -> str:
    """
    Lists all patients who have an appointment record with the currently logged-in doctor.
    supports pagination. 

    Args:
        user_id (Annotated[int, InjectedState): id of the logged-in doctor 
        page (Optional[int], optional): the page number to retrieve starting from 1, Defaults to 1
        page_size (Optional[int], optional): the number of patients to retrieve per page, Defaults to 10
    """
    
    logger.info(f"Tool 'list_my_patients' invoked by doctor_id '{user_id}' with page {page}, page_size {page_size}")
    
    current_page = page if page and page > 0 else 1
    current_page_size = page_size if page_size and page_size > 0 else 10
    offset = (current_page -1) * current_page_size
    
    async with tool_db_session() as db:
        try:
            # user_id here is the requesting_doctor_id
            patients = await get_patients_for_doctor(
                db, requesting_doctor_id=user_id, limit=current_page_size, offset=offset
            )
            
            if not patients:
                if current_page == 1:
                    return "You dont have any patients with appointmnent records in the system"
                else:
                    return "No more patients found for the given page"
            
            response_lines = [f"Listing your patients (Page {current_page}):"]
            for p_idx, patient in enumerate(patients):
                dob_str = patient.dob.strftime('%Y-%m-%d') if patient.dob else 'N/A'
                # Using patient.user_id  as an identifier in the list for now
                response_lines.append(
                    f"{offset + p_idx + 1}. {patient.first_name} { patient.last_name} (ID: {patient.user_id}), DOB: {dob_str}"
                )
            
            if len(patients) < current_page_size:
                response_lines.append("\n(End of list)")
            else:
                response_lines.append(f"\n (Showing {len(patients)} patients. To see more, ask for page {current_page + 1})")
            
            return "\n".join(response_lines)
        except Exception as e:
            logger.error(
                f"Tool: 'list_my_patients: Error processing request for doctor_id '{user_id}': {e}",
                exc_info=True
            )
            return "An unexpected error occurred while trying to retrieve your patient list. Please try again later."