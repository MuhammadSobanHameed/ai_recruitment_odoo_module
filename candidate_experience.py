from odoo import fields,api,models

class CandidateExperience(models.Model):
    _name="candidate.experience"
    _description = "Candidate Experience"

    candidate_id = fields.Many2one('candidate.profile','Candidate', ondelete='cascade')
    sequence = fields.Integer(string="Sequence", default=10)
    job_titles=fields.Char(string="Job Title")
    companies=fields.Char(string="Companies")
    employment_dates=fields.Char(string="Employment Dates")
    key_responsibilities=fields.Text(string="Responsibilities")
