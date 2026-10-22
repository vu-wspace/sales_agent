import json
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool

CAR_INVENTORY: List[Dict[str, Any]] = [
    {
        "id": "CAR001",
        "make": "Toyota",
        "model": "Camry",
        "year": 2024,
        "type": "Sedan",
        "price": 28500,
        "mileage": 0,
        "fuel": "Hybrid",
        "color": "Silver Metallic",
        "features": ["Adaptive Cruise Control", "Lane Departure Warning", "Apple CarPlay", "Heated Seats"]
    },
    {
        "id": "CAR002",
        "make": "Honda",
        "model": "CR-V",
        "year": 2023,
        "type": "SUV",
        "price": 32000,
        "mileage": 12000,
        "fuel": "Gasoline",
        "color": "Crystal Black Pearl",
        "features": ["All-Wheel Drive", "Sunroof", "Blind Spot Information", "Leather Seats"]
    },
    {
        "id": "CAR003",
        "make": "Tesla",
        "model": "Model 3",
        "year": 2024,
        "type": "Sedan",
        "price": 38990,
        "mileage": 0,
        "fuel": "Electric",
        "color": "Pearl White",
        "features": ["Autopilot", "15-inch Touchscreen", "Glass Roof", "Heated Steering Wheel"]
    },
    {
        "id": "CAR004",
        "make": "Ford",
        "model": "F-150",
        "year": 2022,
        "type": "Truck",
        "price": 41500,
        "mileage": 28000,
        "fuel": "Gasoline",
        "color": "Oxford White",
        "features": ["4x4 Drive", "Towing Package", "Trailer Hitch Assist", "Ford Co-Pilot360"]
    },
    {
        "id": "CAR005",
        "make": "BMW",
        "model": "X5",
        "year": 2024,
        "type": "Luxury SUV",
        "price": 68000,
        "mileage": 3500,
        "fuel": "Hybrid",
        "color": "Phytonic Blue",
        "features": ["Panoramic Sky Lounge", "Harman Kardon Audio", "Head-Up Display", "Parking Assistant Plus"]
    }
]


def _find_car(car_id: str) -> Optional[Dict[str, Any]]:
    car_id = car_id.strip().lower()
    return next((c for c in CAR_INVENTORY if c["id"].lower() == car_id), None)


@tool
def search_cars(
    max_price: Optional[float] = None,
    car_type: Optional[str] = None,
    make: Optional[str] = None,
    fuel: Optional[str] = None
) -> str:
    """Search dealership inventory by max budget, body type (SUV, Sedan, Truck), make/brand, or fuel type (Hybrid, Electric, Gasoline)."""
    results = CAR_INVENTORY
    if max_price is not None:
        results = [c for c in results if c["price"] <= max_price]
    if car_type:
        results = [c for c in results if car_type.lower() in c["type"].lower()]
    if make:
        results = [c for c in results if make.lower() in c["make"].lower()]
    if fuel:
        results = [c for c in results if fuel.lower() in c["fuel"].lower()]

    if not results:
        return "No vehicles matched your search criteria. Try adjusting the price range, brand, or vehicle type."

    return json.dumps(results, indent=2)


@tool
def get_car_details(car_id: str) -> str:
    """Retrieve full details, specifications, and features for a specific car by its ID (e.g., CAR001)."""
    car = _find_car(car_id)
    if car is None:
        return f"Car with ID '{car_id}' was not found in our current inventory."
    return json.dumps(car, indent=2)


@tool
def book_test_drive(car_id: str, customer_name: str, preferred_date: str, phone: str) -> str:
    """Schedule a dealership visit / test drive appointment for a customer on a specific car (by ID, e.g., CAR001)."""
    car = _find_car(car_id)
    if car is None:
        return f"Cannot book a test drive: car with ID '{car_id}' was not found. Use search_cars to find a valid ID."
    return (
        f"Test drive successfully scheduled!\n"
        f"- Vehicle: {car['year']} {car['make']} {car['model']} ({car['id']})\n"
        f"- Customer: {customer_name}\n"
        f"- Preferred Date/Time: {preferred_date}\n"
        f"- Contact Phone: {phone}\n"
        f"A sales specialist will call to confirm 1 hour prior to arrival."
    )


tools =[search_cars, get_car_details, book_test_drive]
