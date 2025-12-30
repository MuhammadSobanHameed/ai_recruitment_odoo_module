from odoo import models, fields, api

class UserInherit(models.Model):
    _inherit = 'res.users'
    recruitment_consultants = fields.Boolean(default=False)
