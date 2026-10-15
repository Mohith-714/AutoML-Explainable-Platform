"""Centralized enterprise SaaS design system tokens, CSS styles, and UI components."""

from __future__ import annotations

import streamlit as st

# SaaS Design Tokens
COLOR_BG = "#F5F7FB"
COLOR_SURFACE = "#FFFFFF"
COLOR_BORDER = "#E2E8F0"
COLOR_TEXT_PRIMARY = "#172033"
COLOR_TEXT_SECONDARY = "#667085"
COLOR_TEXT_MUTED = "#94A3B8"
COLOR_PRIMARY_BLUE = "#2563EB"
COLOR_PRIMARY_HOVER = "#1D4ED8"
COLOR_PRIMARY_PALE = "#EFF6FF"
COLOR_SUCCESS = "#059669"
COLOR_WARNING = "#D97706"
COLOR_DANGER = "#DC2626"

SAAS_CSS = """
<style>
/* Import Inter Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Base Typography & App Layout */
html, body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color: #172033 !important;
    background-color: #F5F7FB !important;
}

.stApp {
    background-color: #F5F7FB !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1250px !important;
}

/* Base Headings & Captions */
h1, h2, h3, h4, h5, h6, [data-testid="stHeadingWithActionElements"] {
    color: #172033 !important;
    font-weight: 600 !important;
    letter-spacing: -0.015em !important;
}

.stCaption, small, [data-testid="stCaptionContainer"] p {
    color: #667085 !important;
    font-size: 13px !important;
}

/* ========================================================================= */
/* 1. SIDEBAR NAVIGATION STYLING (FIX TEXT VISIBILITY & REMOVE RADIO DOTS)   */
/* ========================================================================= */

section[data-testid="stSidebar"],
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
    box-shadow: none !important;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.25rem !important;
    padding-left: 0.85rem !important;
    padding-right: 0.85rem !important;
}

/* Hide default circular radio dots completely */
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label input[type="radio"] {
    display: none !important;
}

/* Style radio group into vertical SaaS nav list */
[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 4px !important;
    display: flex !important;
    flex-direction: column !important;
}

/* Nav Item Container */
[data-testid="stSidebar"] div[role="radiogroup"] label {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    padding: 10px 14px !important;
    border-radius: 8px !important;
    margin-bottom: 2px !important;
    cursor: pointer !important;
    background-color: transparent !important;
    border-left: 3px solid transparent !important;
    transition: all 0.15s ease-in-out !important;
}

/* Explicit dark text color for inactive navigation labels */
[data-testid="stSidebar"] div[role="radiogroup"] label,
[data-testid="stSidebar"] div[role="radiogroup"] label p,
[data-testid="stSidebar"] div[role="radiogroup"] label span,
[data-testid="stSidebar"] div[role="radiogroup"] label div {
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #344054 !important;
    line-height: 1.4 !important;
    margin: 0 !important;
}

/* Inactive Hover state */
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background-color: #F8FAFC !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover p,
[data-testid="stSidebar"] div[role="radiogroup"] label:hover span,
[data-testid="stSidebar"] div[role="radiogroup"] label:hover div {
    color: #172033 !important;
}

/* Active / Selected state: pale blue background (#EFF6FF), dark blue text (#2563EB), 3px blue border (#2563EB) */
[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background-color: #EFF6FF !important;
    border-left: 3px solid #2563EB !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p,
[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] span,
[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] div,
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p,
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) span,
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) div {
    color: #2563EB !important;
    font-weight: 600 !important;
}

/* Focus accessible ring */
[data-testid="stSidebar"] div[role="radiogroup"] label:focus-within {
    outline: 2px solid #93C5FD !important;
    outline-offset: 1px !important;
}

/* Line Icons injected on Sidebar navigation items via ::before */
div[data-testid="stSidebar"] div[role="radiogroup"] label::before {
    content: "";
    display: inline-block;
    width: 18px;
    height: 18px;
    margin-right: 10px;
    flex-shrink: 0;
    background-repeat: no-repeat;
    background-position: center;
    background-size: contain;
}

/* Inactive Icon States */
div[data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(1)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%2364748B' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z'/%3E%3Cpolyline points='9 22 9 12 15 12 15 22'/%3E%3C/svg%3E");
}
div[data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(2)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%2364748B' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cellipse cx='12' cy='5' rx='9' ry='3'/%3E%3Cpath d='M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5'/%3E%3Cpath d='M3 12c0 1.66 4 3 9 3s9-1.34 9-3'/%3E%3C/svg%3E");
}
div[data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(3)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%2364748B' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='4' x2='20' y1='21' y2='21'/%3E%3Cline x1='4' x2='20' y1='14' y2='14'/%3E%3Cline x1='4' x2='20' y1='7' y2='7'/%3E%3Ccircle cx='8' cy='21' r='2'/%3E%3Ccircle cx='16' cy='14' r='2'/%3E%3Ccircle cx='10' cy='7' r='2'/%3E%3C/svg%3E");
}
div[data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(4)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%2364748B' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M10 2v7.527a2 2 0 0 1-.211.896L4.72 20.55a1 1 0 0 0 .9 1.45h12.76a1 1 0 0 0 .9-1.45l-5.069-10.127A2 2 0 0 1 14 9.527V2'/%3E%3Cline x1='8.5' x2='15.5' y1='2' y2='2'/%3E%3C/svg%3E");
}
div[data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(5)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%2364748B' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='12' x2='12' y1='20' y2='10'/%3E%3Cline x1='18' x2='18' y1='20' y2='4'/%3E%3Cline x1='6' x2='6' y1='20' y2='16'/%3E%3C/svg%3E");
}
div[data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(6)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%2364748B' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5'/%3E%3Cpath d='M9 18h6'/%3E%3Cpath d='M10 22h4'/%3E%3C/svg%3E");
}
div[data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(7)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%2364748B' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M22 12h-4l-3 9L9 3l-3 9H2'/%3E%3C/svg%3E");
}
div[data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(8)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%2364748B' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m21 16-7.85 4.5a2 2 0 0 1-2.3 0L3 16V8a2 2 0 0 1 1-1.73l7-4a2 2 0 0 1 2 0l7 4A2 2 0 0 1 21 8z'/%3E%3Cpolyline points='3.27 6.96 12 12.01 20.73 6.96'/%3E%3Cline x1='12' x2='12' y1='22.08' y2='12'/%3E%3C/svg%3E");
}
div[data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(9)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%2364748B' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z'/%3E%3Cpath d='M14 2v4a2 2 0 0 0 2 2h4'/%3E%3Cpath d='M10 9H8'/%3E%3Cpath d='M16 13H8'/%3E%3Cpath d='M16 17H8'/%3E%3C/svg%3E");
}
div[data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(10)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%2364748B' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='3'/%3E%3Cpath d='M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z'/%3E%3C/svg%3E");
}
div[data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(11)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%2364748B' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpath d='M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3'/%3E%3Cpath d='M12 17h.01'/%3E%3C/svg%3E");
}

/* Active Icon States (Blue Stroke %232563EB) */
div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"]:nth-child(1)::before,
div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked):nth-child(1)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%232563EB' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z'/%3E%3Cpolyline points='9 22 9 12 15 12 15 22'/%3E%3C/svg%3E");
}
div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"]:nth-child(2)::before,
div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked):nth-child(2)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%232563EB' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cellipse cx='12' cy='5' rx='9' ry='3'/%3E%3Cpath d='M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5'/%3E%3Cpath d='M3 12c0 1.66 4 3 9 3s9-1.34 9-3'/%3E%3C/svg%3E");
}
div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"]:nth-child(3)::before,
div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked):nth-child(3)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%232563EB' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='4' x2='20' y1='21' y2='21'/%3E%3Cline x1='4' x2='20' y1='14' y2='14'/%3E%3Cline x1='4' x2='20' y1='7' y2='7'/%3E%3Ccircle cx='8' cy='21' r='2'/%3E%3Ccircle cx='16' cy='14' r='2'/%3E%3Ccircle cx='10' cy='7' r='2'/%3E%3C/svg%3E");
}
div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"]:nth-child(4)::before,
div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked):nth-child(4)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%232563EB' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M10 2v7.527a2 2 0 0 1-.211.896L4.72 20.55a1 1 0 0 0 .9 1.45h12.76a1 1 0 0 0 .9-1.45l-5.069-10.127A2 2 0 0 1 14 9.527V2'/%3E%3Cline x1='8.5' x2='15.5' y1='2' y2='2'/%3E%3C/svg%3E");
}
div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"]:nth-child(5)::before,
div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked):nth-child(5)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%232563EB' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='12' x2='12' y1='20' y2='10'/%3E%3Cline x1='18' x2='18' y1='20' y2='4'/%3E%3Cline x1='6' x2='6' y1='20' y2='16'/%3E%3C/svg%3E");
}
div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"]:nth-child(6)::before,
div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked):nth-child(6)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%232563EB' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5'/%3E%3Cpath d='M9 18h6'/%3E%3Cpath d='M10 22h4'/%3E%3C/svg%3E");
}
div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"]:nth-child(7)::before,
div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked):nth-child(7)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%232563EB' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M22 12h-4l-3 9L9 3l-3 9H2'/%3E%3C/svg%3E");
}
div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"]:nth-child(8)::before,
div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked):nth-child(8)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%232563EB' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m21 16-7.85 4.5a2 2 0 0 1-2.3 0L3 16V8a2 2 0 0 1 1-1.73l7-4a2 2 0 0 1 2 0l7 4A2 2 0 0 1 21 8z'/%3E%3Cpolyline points='3.27 6.96 12 12.01 20.73 6.96'/%3E%3Cline x1='12' x2='12' y1='22.08' y2='12'/%3E%3C/svg%3E");
}
div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"]:nth-child(9)::before,
div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked):nth-child(9)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%232563EB' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z'/%3E%3Cpath d='M14 2v4a2 2 0 0 0 2 2h4'/%3E%3Cpath d='M10 9H8'/%3E%3Cpath d='M16 13H8'/%3E%3Cpath d='M16 17H8'/%3E%3C/svg%3E");
}
div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"]:nth-child(10)::before,
div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked):nth-child(10)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%232563EB' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='3'/%3E%3Cpath d='M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z'/%3E%3C/svg%3E");
}
div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"]:nth-child(11)::before,
div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked):nth-child(11)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%232563EB' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpath d='M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3'/%3E%3Cpath d='M12 17h.01'/%3E%3C/svg%3E");
}

/* ========================================================================= */
/* 2. CARD CONTAINERS (FIX BLANK CARDS VIA NATIVE BORDERED WRAPPER)         */
/* ========================================================================= */

/* Streamlit Native Bordered Containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 22px 24px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.02) !important;
    margin-bottom: 16px !important;
}

/* Standalone HTML SaaS Surface Cards */
.saas-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 22px 24px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.02);
    margin-bottom: 16px;
}

.card-header-flex {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 14px;
}

.card-title {
    font-size: 16px;
    font-weight: 600;
    color: #172033;
    margin: 0;
}

.card-subtitle {
    font-size: 13px;
    color: #667085;
    margin-top: 3px;
    margin-bottom: 0;
}

/* Metric Summary Cards */
.summary-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px 22px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.summary-card-top {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
}

.summary-icon-box {
    width: 36px;
    height: 36px;
    border-radius: 8px;
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #475569;
}

.summary-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #667085;
}

.summary-value {
    font-size: 20px;
    font-weight: 700;
    color: #172033;
    line-height: 1.2;
    margin-bottom: 6px;
}

.summary-subtext {
    font-size: 13px;
    color: #667085;
}

/* Badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
}
.badge-success { background-color: #ECFDF5; color: #059669; border: 1px solid #A7F3D0; }
.badge-danger { background-color: #FEF2F2; color: #DC2626; border: 1px solid #FECACA; }
.badge-warning { background-color: #FFFBEB; color: #D97706; border: 1px solid #FDE68A; }
.badge-neutral { background-color: #F1F5F9; color: #667085; border: 1px solid #E2E8F0; }
.badge-blue { background-color: #EFF6FF; color: #2563EB; border: 1px solid #BFDBFE; }

/* 5-Stage Lifecycle Stepper */
.stepper-container {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    position: relative;
    padding: 16px 0;
    gap: 12px;
}

.stepper-step {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    position: relative;
    z-index: 2;
}

.stepper-icon-circle {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 12px;
    color: #334155;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

.stepper-title {
    font-size: 14px;
    font-weight: 600;
    color: #172033;
    margin-bottom: 4px;
}

.stepper-desc {
    font-size: 12px;
    color: #667085;
    line-height: 1.4;
    max-width: 130px;
}

.stepper-line {
    position: absolute;
    top: 38px;
    left: 8%;
    right: 8%;
    height: 2px;
    background-color: #E2E8F0;
    z-index: 1;
}

/* Getting Started Checklist Item */
.checklist-item {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    margin-bottom: 18px;
}

.checklist-num {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background-color: #EFF6FF;
    color: #2563EB;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 600;
    flex-shrink: 0;
}

.checklist-text-title {
    font-size: 14px;
    font-weight: 600;
    color: #172033;
    margin-bottom: 2px;
}

.checklist-text-desc {
    font-size: 12px;
    color: #667085;
}

/* Empty State Box */
.empty-state-box {
    text-align: center;
    padding: 36px 20px;
    color: #667085;
}

.empty-state-title {
    font-size: 15px;
    font-weight: 600;
    color: #172033;
    margin-top: 10px;
    margin-bottom: 4px;
}

.empty-state-text {
    font-size: 13px;
    color: #667085;
    max-width: 380px;
    margin: 0 auto;
    line-height: 1.5;
}

/* Native Button & Widget Overrides */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    transition: all 0.15s ease-in-out !important;
}

.stButton > button[kind="primary"] {
    background-color: #2563EB !important;
    border-color: #2563EB !important;
    color: #FFFFFF !important;
    box-shadow: 0 1px 2px rgba(37, 99, 235, 0.2) !important;
}

.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span {
    color: #FFFFFF !important;
}

.stButton > button[kind="primary"]:hover {
    background-color: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
}

.stButton > button[kind="secondary"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    color: #172033 !important;
}

.stButton > button[kind="secondary"]:hover {
    background-color: #F8FAFC !important;
    border-color: #CBD5E1 !important;
}

/* Form Widget Labels */
label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {
    color: #172033 !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}

/* Clean Dataframe Table Styling */
div[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    background-color: #FFFFFF !important;
}

/* Native Metrics Styling */
[data-testid="stMetricValue"] {
    color: #172033 !important;
    font-weight: 700 !important;
}

[data-testid="stMetricLabel"] {
    color: #667085 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
}

/* Tab Bar Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 16px;
    border-bottom: 1px solid #E2E8F0;
}

.stTabs [data-baseweb="tab"] {
    font-size: 14px;
    font-weight: 500;
    color: #667085;
    padding: 10px 4px;
}

.stTabs [aria-selected="true"] {
    color: #2563EB !important;
    font-weight: 600 !important;
    border-bottom-color: #2563EB !important;
}
</style>
"""

# Lucide-style SVG Icons
ICONS = {
    "logo": """<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>""",
    "database": """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>""",
    "cube": """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>""",
    "activity": """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>""",
    "clock": """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>""",
    "settings": """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>""",
    "chart": """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>""",
    "shield": """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>""",
    "document": """<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>""",
    "upload": """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>""",
}


def apply_theme() -> None:
    """Inject global SaaS stylesheet."""
    st.markdown(SAAS_CSS, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str) -> None:
    """Render standardized top title block with active dataset and target indicators."""
    dataset_name = st.session_state.get("dataset_name")
    target_col = st.session_state.get("target_column")
    task_type = st.session_state.get("task_type")
    features = st.session_state.get("selected_features")
    dataset = st.session_state.get("dataset")

    if dataset is not None and dataset_name:
        row_count = f"{len(dataset):,} rows"
        feat_count = f"{len(features)} features" if features else f"{len(dataset.columns)-1 if target_col else len(dataset.columns)} features"
        target_badge = f'<span style="background: #eff6ff; color: #1d4ed8; padding: 2px 8px; border-radius: 4px; font-weight: 600;">🎯 Target: {target_col}</span>' if target_col else '<span style="background: #fef2f2; color: #b91c1c; padding: 2px 8px; border-radius: 4px; font-weight: 600;">⚠️ Target: Not selected</span>'
        task_str = task_type.title() if task_type else ("Classification" if target_col else "Pending")
        task_badge = f'<span style="background: #f1f5f9; color: #334155; padding: 2px 8px; border-radius: 4px; font-weight: 500;">⚙️ Task: {task_str}</span>'
        status_bar_html = f"""
        <div style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-top: 10px; font-size: 13px; color: #475467;">
            <span style="font-weight: 600; color: #1e293b;">📁 {dataset_name} ({row_count})</span>
            {target_badge}
            {task_badge}
            <span style="background: #f8fafc; color: #64748b; padding: 2px 8px; border-radius: 4px; border: 1px solid #e2e8f0;">🧩 {feat_count}</span>
        </div>
        """
    else:
        status_bar_html = """
        <div style="display: flex; gap: 8px; align-items: center; margin-top: 8px; font-size: 13px; color: #94a3b8;">
            <span>📁 Active Dataset: None loaded</span>
        </div>
        """

    st.markdown(
        f"""
        <div style="margin-bottom: 22px;">
            <h1 style="font-size: 28px; font-weight: 700; color: #172033; margin: 0; letter-spacing: -0.02em;">{title}</h1>
            <p style="font-size: 14px; color: #667085; margin-top: 4px; margin-bottom: 0;">{subtitle}</p>
            {status_bar_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
