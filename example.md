# Submission Example

This is an example of how you could contribute knowledge and skills around FedRamp.  It covers the addition of knowledge in the form of material
from the official [FedRamp](fedramp.gov) site.  It also covers adding knowledge about [FIPS](https://www.nist.gov/standardsgov/compliance-faqs-federal-information-processing-standards-fips).
Lastly, it covers adding specific knowledge about Red Hat FIPS and FedRamp certifications.

After this knowledge has been added, you likely want to add a skill.  This comes in the form of question and answer suggestions.  In this example,
we ask questions related to the knowledge sections we added and expected answers.  This will then be augmented with sythentic data to learn this
skill.

The end result is with a simple addition of knowledge and a few examples of the skill that needs to be learned, you can train the foundation model
in a new area.

## Knowledge

The knowledge must be contributed in a [Markdown](https://www.markdownguide.org/basic-syntax/) format.  But more importantly, you only want to add
content that is covered by questions and answers.  So in our case, while the FedRamp documents are hundreds of pages in length, we will only add
small sections that we will cover will questions and answers.

### Steps

The first step is to obtain the publicly available documentation (i.e. knowledge) that we are going to work with.  In this case, we will start by
providing knowledge around all the [Red Hat Cryptographic Module Validation Program (CMVP) certifications](https://csrc.nist.gov/projects/cryptographic-module-validation-program/validated-modules/search?SearchMode=Basic&Vendor=Red+Hat&CertificateStatus=Active&ValidationYear=0).
We need to convert the table from the results into Markdown.  A simple manual conversion yields the following results (look at the raw source to see the Markdown formatted content):

| Certificate Number |	Vendor Name |	Module Name |	Module Type	| Validation Date |
| ------------------ | ------------ | ----------- | ----------- | --------------- |
| 4642 | Red Hat®, Inc. | Red Hat Enterprise Linux 8 OpenSSL Cryptographic Module	| Software | 10/25/2023 |
| 4498 | Red Hat®, Inc.	| Red Hat Enterprise Linux 7 NSS Cryptographic Module	| Software | 05/02/2023 |
| 4458 | Red Hat®, Inc.	| Red Hat Enterprise Linux 8 NSS Cryptographic Module	| Software | 03/24/2023, 10/13/2023 |
| 4438 | Red Hat®, Inc.	| Red Hat Enterprise Linux 8 libgcrypt Cryptographic Module	| Software | 02/15/2023, 10/27/2023 |
| 4434 | Red Hat®, Inc.	| Red Hat Enterprise Linux 8 Kernel Crypto API Cryptographic Module	| Software | 02/15/2023 |
| 4428 | Red Hat®, Inc.	| Red Hat Enterprise Linux 8 GnuTLS Cryptographic Module | Software | 01/27/2023, 10/13/2023 |
| 4413 | Red Hat®, Inc.	| Red Hat Enterprise Linux 8 NSS Cryptographic Module	| Software | 01/16/2023 |
| 4397 | Red Hat®, Inc.	| Red Hat Enterprise Linux 8 libgcrypt Cryptographic Module	| Software | 12/20/2022 |
| 4384 | Red Hat®, Inc.	| Red Hat Enterprise Linux 8 Kernel Crypto API Cryptographic Module	| Software | 12/05/2022 |
| 4272 | Red Hat®, Inc.	| Red Hat Enterprise Linux 8 GnuTLS Cryptographic Module | Software | 07/18/2022 |
| 4271 | Red Hat®, Inc.	| Red Hat Enterprise Linux 8 OpenSSL Cryptographic Module	| Software | 07/18/2022 |
| 4254 | Red Hat®, Inc.	| Red Hat Enterprise Linux 8 Kernel Crypto API Cryptographic Module | Software | 06/23/2022 |
| 3939 | Red Hat®, Inc.	| Red Hat Enterprise Linux 7 Kernel Crypto API Cryptographic Module | Software | 06/01/2021, 10/01/2021, 10/06/2021 |
| 3784 | Red Hat®, Inc.	| Red Hat Enterprise Linux 8 libgcrypt Cryptographic Module	| Software | 12/21/2020 |

We save this Markdown table off into it's own file.

The next step is to provide knowledge about the [SSP Appendix Q - Cryptographic Modules Table](https://www.fedramp.gov/assets/resources/templates/SSP-Appendix-Q-Cryptographic-Modules-Table.docx) table for a FedRamp submission.
This is one of the many [resources](https://www.fedramp.gov/documents-templates/) involved in a
FedRamp submission.

Since the download is a word document, we will use pandoc to convert the word document into Markdown format:

> pandoc -t markdown ~/Downloads/SSP-Appendix-Q-Cryptographic-Modules-Table.docx > SSP-Appendix-Q-Example.md

## Core Skill

Now we are going to teach our model how to fill out the FedRamp [SSP Appendix Q - Cryptographic Modules Table](https://www.fedramp.gov/assets/resources/templates/SSP-Appendix-Q-Cryptographic-Modules-Table.docx) table for a FedRamp submission
that is utilizing RHEL 7 or RHEL 8.  The goal of this skill would be to make it easier to utilize this model to compose a FedRamp submission when
utilizing Red Hat technology.

In this case, we will be providing question and answer examples 
