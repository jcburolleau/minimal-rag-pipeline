"""Render entity YAML into markdown documents with YAML frontmatter."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from app.config import settings  # noqa: E402

FICTION_BANNER = (
    "This document is part of a **fictional** knowledge base for Northline "
    "Sensors SA. It is not a real company, product, or contract."
)


def load_yaml(name: str):
    path = settings.entities_path / name
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_doc(doc_id: str, title: str, doc_type: str, date: str, body: str) -> None:
    settings.docs_path.mkdir(parents=True, exist_ok=True)
    path = settings.docs_path / f"{doc_id}.md"
    front = "\n".join(
        [
            "---",
            f"id: {doc_id}",
            f"title: {title}",
            f"doc_type: {doc_type}",
            f"date: {date}",
            "---",
            "",
        ]
    )
    path.write_text(front + body.strip() + "\n", encoding="utf-8")


def money(value: int | float, currency: str = "CHF") -> str:
    if isinstance(value, float):
        return f"{currency} {value:,.2f}"
    return f"{currency} {value:,}".replace(",", " ")


def render_company(company: dict) -> None:
    units = "\n".join(
        f"- **{unit['name']}**, lead {unit['lead']}" for unit in company["business_units"]
    )
    body = f"""
# {company['name']}

{FICTION_BANNER}

## Profile

{company['name']} ({company['legal_name']}) was founded on {company['founded']}
and is headquartered at {company['hq']}. VAT number {company['vat']}.
The company employs {company['employees']} people.

## Leadership

- CEO (Chief Executive Officer): {company['ceo']}
- CFO (Chief Financial Officer): {company['cfo']}
- CTO (Chief Technology Officer): {company['cto']}

## Mission

{company['mission'].strip()}

## Business units

{units}

## Support

Support is at {company['support']['email']}, {company['support']['hours']}.
Standard hardware warranty is {company['support']['warranty_years']} years.

## Banking

Invoices are paid in {company['banking']['currency']} to IBAN {company['banking']['iban']}.

## Fictional notice

{company['fictional_notice'].strip()}
"""
    dump_doc("company-overview", f"{company['name']} overview", "company", company["founded"], body)


def render_products(products: list[dict]) -> None:
    for product in products:
        bom = "\n".join(
            f"- **{item['part']}** — {item['name']} (supplier: {item['supplier']})"
            for item in product["bom"]
        )
        spec_lines = []
        for key, value in product["specs"].items():
            label = key.replace("_", " ")
            spec_lines.append(f"- {label}: {value}")
        specs = "\n".join(spec_lines)
        body = f"""
# {product['sku']} {product['name']}

{FICTION_BANNER}

## Overview

SKU **{product['sku']}**. Category: {product['category']}.
{product['summary'].strip()}

## Commercial terms

- List price: {money(product['unit_price_chf'])}
- Standard lead time: {product['lead_time_days']} days
- Price list date: {product['list_date']}

## Specifications

{specs}

## Bill of materials

{bom}

## Notes

{product['notes'].strip()}
"""
        dump_doc(
            f"product-{product['sku'].lower()}",
            f"{product['sku']} {product['name']}",
            "product",
            product["list_date"],
            body,
        )


def render_suppliers(suppliers: list[dict]) -> None:
    for supplier in suppliers:
        body = f"""
# {supplier['name']}

{FICTION_BANNER}

## Profile

{supplier['name']} is a {supplier['category'].lower()} supplier in
{supplier['city']}, {supplier['country']}. Northline has bought from them since
{supplier['since']}.

## Commercial

- Typical lead time: {supplier['lead_time_days']} days
- Payment terms: {supplier['payment_terms']}
- Contact: {supplier['contact']} ({supplier['email']})

## Notes

{supplier['notes'].strip()}
"""
        dump_doc(
            f"supplier-{supplier['id']}",
            supplier["name"],
            "supplier",
            supplier["since"],
            body,
        )


def render_contracts(payload: list[dict]) -> None:
    for contract in payload:
        exclusive = ""
        if contract.get("exclusive_parts"):
            parts = ", ".join(contract["exclusive_parts"])
            exclusive = f"\n- Exclusive parts: {parts}"
        extra = ""
        if contract.get("part"):
            extra += f"\n- Covered part: {contract['part']}"
        if contract.get("lead_time_days"):
            extra += f"\n- Current lead time: {contract['lead_time_days']} days"
        if contract.get("original_lead_time_days"):
            extra += (
                f"\n- Original lead time: {contract['original_lead_time_days']} days "
                "(superseded if an amendment exists)"
            )
        if contract.get("unit_price_eur"):
            extra += f"\n- Piece price: EUR {contract['unit_price_eur']:.2f}"
        body = f"""
# {contract['title']}

{FICTION_BANNER}

## Parties and term

Agreement **{contract['id']}** with {contract['supplier_name']}.
Signed {contract['signed']}. Term {contract['start']} to **{contract['end']}**.

## Commercial terms

- Annual volume cap: {money(contract['annual_cap_chf'])}
- Liability cap: {money(contract['liability_cap_chf'])}
- Incoterms: {contract['incoterms']}
- Payment terms: {contract['payment_terms']}
- Currency: {contract['currency']}{exclusive}{extra}

## Renewal

{contract['renewal']}

## Notes

{contract['notes'].strip()}
"""
        dump_doc(
            f"contract-{contract['id'].lower()}",
            contract["title"],
            "contract",
            contract["signed"],
            body,
        )


def render_amendments(amendments: list[dict]) -> None:
    for amendment in amendments:
        changes = "\n".join(
            f"- {change['field']}: {change['from']} → {change['to']}"
            for change in amendment["changes"]
        )
        body = f"""
# {amendment['title']}

{FICTION_BANNER}

## Effect

Amendment **{amendment['id']}** amends agreement **{amendment['amends']}**.
Signed {amendment['signed']}. Effective **{amendment['effective']}**.

When this amendment and the original agreement disagree, prefer this document.
It is newer.

## Summary

{amendment['summary'].strip()}

## Changed clauses

{changes}

## Operational impact

{amendment['impact'].strip()}
"""
        dump_doc(
            f"amendment-{amendment['id'].lower()}",
            amendment["title"],
            "amendment",
            amendment["signed"],
            body,
        )


def render_policies(policies: list[dict]) -> None:
    for policy in policies:
        body = f"""
# {policy['title']}

{FICTION_BANNER}

## Summary

Policy **{policy['id']}**, owner {policy['owner']}. Dated {policy['date']}.
{policy['summary']}

## Policy text

{policy['body'].strip()}
"""
        dump_doc(
            f"policy-{policy['id'].lower()}",
            policy["title"],
            "policy",
            policy["date"],
            body,
        )


def render_memos(memos: list[dict]) -> None:
    for memo in memos:
        body = f"""
# {memo['title']}

{FICTION_BANNER}

## Header

Memo **{memo['id']}** from {memo['author']} to {memo['audience']}.
Date **{memo['date']}**.

## Body

{memo['body'].strip()}
"""
        dump_doc(
            f"memo-{memo['id'].lower()}",
            memo["title"],
            "memo",
            memo["date"],
            body,
        )


def main() -> None:
    if settings.docs_path.exists():
        for stale in settings.docs_path.glob("*.md"):
            stale.unlink()

    company = load_yaml("company.yaml")
    products = load_yaml("products.yaml")
    suppliers = load_yaml("suppliers.yaml")
    contracts_file = load_yaml("contracts.yaml")
    policies = load_yaml("policies.yaml")
    memos = load_yaml("memos.yaml")

    contracts = contracts_file["agreements"]
    amendments = contracts_file.get("amendments") or []

    render_company(company)
    render_products(products)
    render_suppliers(suppliers)
    render_contracts(contracts)
    render_amendments(amendments)
    render_policies(policies)
    render_memos(memos)

    count = len(list(settings.docs_path.glob("*.md")))
    print(f"Wrote {count} markdown documents to {settings.docs_path}")


if __name__ == "__main__":
    main()
