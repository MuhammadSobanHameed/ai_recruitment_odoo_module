from odoo import models, fields, api


class FunctionalExperience(models.Model):
    _name = 'functional.experience'
    _description = 'Functional Experience'

    name = fields.Char(string='Name')
    code = fields.Char(string='Code')


class SubExperience(models.Model):
    _name = 'sub.experience'
    _description = 'Sub Experience'

    name = fields.Char(string='Name')
    code = fields.Char(string='Code')
