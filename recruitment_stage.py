# -*- coding: utf-8 -*-
from odoo import api, fields, models, _


class RecruitmentStageInherit(models.Model):
    _inherit = "hr.recruitment.stage"

    # Add new fields for assessment stage
    is_assessment_stage = fields.Boolean(
        string='Assessment Stage',
        help="If checked, this stage is used for candidate assessment"
    )
    assessment_type = fields.Selection([
        ('online', 'Online Test'),
        ('physical', 'Physical Test at Office')
    ], string='Assessment Type', help="Type of assessment for this stage")
    
    online_test_template_id = fields.Many2one(
        'mail.template', 
        string="Online Test Email Template",
        domain="[('model', '=', 'hr.applicant')]",
        help="Email template for online test with attachment"
    )
    
    physical_test_template_id = fields.Many2one(
        'mail.template',
        string="Physical Test Email Template", 
        domain="[('model', '=', 'hr.applicant')]",
        help="Email template for physical test invitation"
    )